import os
import sys
import uuid
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory
import torch
from werkzeug.utils import secure_filename
from twilio.rest import Client  # Twilio import for WhatsApp integration
from typing import Optional, List, Dict
import base64
import json
import requests
import mimetypes

# Optional: Gemini (Google Generative AI) for dynamic alert descriptions
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    genai = None
    GEMINI_AVAILABLE = False
from dotenv import load_dotenv
load_dotenv()
try:
    # Also load optional supabase-specific overrides if present
    _alt_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.supabase")
    if os.path.exists(_alt_env):
        load_dotenv(_alt_env, override=True)
except Exception:
    pass

# Add parent directory to path to import from project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules  
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Try to import the models
try:
    from models.advanced_model import MaxAccuracyC3D, AdvancedVideoModel
    MAXACCURACY_MODEL_AVAILABLE = True
except ImportError:
    MaxAccuracyC3D = None
    AdvancedVideoModel = None
    MAXACCURACY_MODEL_AVAILABLE = False

try:
    from models.cnn3d import C3D
    C3D_AVAILABLE = True
except ImportError:
    C3D = None
    C3D_AVAILABLE = False

MODEL_AVAILABLE = MAXACCURACY_MODEL_AVAILABLE or C3D_AVAILABLE

# Import utils directly from local directory
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from video_utils import save_prediction_visualization, export_anomaly_clip

# Now import inference
sys.path.insert(0, current_dir)

# Define predict_video function locally to avoid import issues
def predict_video(model, video_path, clip_len=16, device='cuda'):
    """Predict anomaly class for a video"""
    import cv2
    import numpy as np
    from torchvision import transforms
    from tqdm import tqdm
    
    def extract_frames_from_video_local(video_path, sample_rate=10):
        """Extract frames from a video file"""
        frames = []
        video = cv2.VideoCapture(video_path)
        
        if not video.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames
        
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for i in range(frame_count):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            
            if ret and i % sample_rate == 0:
                frames.append(frame)
            elif not ret:
                break
        
        video.release()
        return frames
    
    def process_video_frames_local(frames, clip_len=16, overlap=0.5):
        """Process extracted frames into clips"""
        if len(frames) < clip_len:
            frames = frames * (clip_len // len(frames) + 1)
            frames = frames[:clip_len]
            return [frames]
        
        step_size = max(1, int(clip_len * (1 - overlap)))
        clips = []
        for i in range(0, len(frames) - clip_len + 1, step_size):
            clip = frames[i:i + clip_len]
            clips.append(clip)
        
        return clips
    
    # Extract frames from video
    frames = extract_frames_from_video_local(video_path)
    
    # Process frames for prediction
    clips = process_video_frames_local(frames, clip_len=clip_len)
    
    # Transform for inference
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Process each clip
    predictions = []
    confidences = []
    
    # tqdm can raise OSError on some Windows/Flask consoles; use a simple loop for web safety
    for clip in clips:
        # Apply transformation to each frame
        processed_clip = []
        for frame in clip:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = transform(frame)
            processed_clip.append(frame)
        
        # Stack frames and reorder dimensions
        processed_clip = torch.stack(processed_clip)
        processed_clip = processed_clip.permute(1, 0, 2, 3)
        processed_clip = processed_clip.unsqueeze(0)
        processed_clip = processed_clip.to(device)
        
        # Get prediction
        try:
            with torch.no_grad():
                outputs = model(processed_clip)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, prediction = probs.max(1)
                
                predictions.append(prediction.item())
                confidences.append(confidence.item())
        except RuntimeError as e:
            print(f"Error during prediction: {e}")
            continue
    
    # Get the most common prediction and average confidence
    if predictions:
        pred_counts = np.bincount(predictions)
        most_common_pred = np.argmax(pred_counts)
        avg_confidence = np.mean([confidences[i] for i, p in enumerate(predictions) if p == most_common_pred])
        return most_common_pred, avg_confidence, predictions, confidences
    else:
        return None, 0, [], []

# Model loading function
def load_model(model_path, num_classes=14, device='cuda'):
    """Load a trained model"""
    model = C3D(num_classes=num_classes)
    if torch.cuda.is_available() and device == 'cuda':
        checkpoint = torch.load(model_path, map_location=device)
        device_obj = torch.device('cuda')
    else:
        checkpoint = torch.load(model_path, map_location='cpu')
        device_obj = torch.device('cpu')
        device = 'cpu'
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device_obj)
    model.eval()
    return model

# Create template directory if it doesn't exist
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

os.makedirs(template_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)
os.makedirs(os.path.join(static_dir, "uploads"), exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Create a basic index.html template if it doesn't exist
index_html_path = os.path.join(template_dir, "index.html")
if not os.path.exists(index_html_path):
    with open(index_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Video Crime Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            border: 2px dashed #ccc;
            padding: 20px;
            text-align: center;
            margin-top: 30px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            margin-top: 10px;
        }
        .message {
            margin: 20px 0;
            padding: 10px;
            border-radius: 5px;
        }
        .error {
            background-color: #f8d7da;
            color: #721c24;
        }
        /* Alert section styles */
        .alert-settings {
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .alert-settings h3 {
            margin-top: 0;
        }
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin: 10px 0;
        }
        .phone-input {
            width: 100%;
            padding: 8px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <h1>Video Crime Detection System</h1>
    
    {% with messages = get_flashed_messages(with_categories=true) %}
        {% if messages %}
            {% for category, message in messages %}
                <div class="message {{ category }}">{{ message }}</div>
            {% endfor %}
        {% endif %}
    {% endwith %}
    
    <div class="upload-form">
        <h2>Upload Video</h2>
        <p>Select a video file to analyze (MP4, AVI, MOV, MKV)</p>
        <form action="{{ url_for('upload_file') }}" method="post" enctype="multipart/form-data">
            <input type="file" name="video" accept=".mp4,.avi,.mov,.mkv">
            
            <!-- WhatsApp Alert Settings - Always enabled with fixed number -->
            <div class="alert-settings">
                <h3>WhatsApp Alert Settings</h3>
                <p>Select which crime types should trigger alerts:</p>
                
                <div class="checkbox-group">
                    {% for class_name in class_names %}
                    <label>
                        <input type="checkbox" name="alert_classes" value="{{ class_name }}" 
                               {% if class_name in ["Shooting", "Robbery", "Explosion", "Assault", "Arson"] %}checked{% endif %}> 
                        {{ class_name }}
                    </label>
                    {% endfor %}
                </div>
                
                <p>WhatsApp alerts will be sent to: +91 7397555299</p>
                <input type="hidden" name="phone_number" value="+917397555299">
            </div>
            
            <br>
            <button type="submit" class="btn">Upload & Analyze</button>
        </form>
    </div>
</body>
</html>
""")

# Create a basic result.html template if it doesn't exist
result_html_path = os.path.join(template_dir, "result.html")
if not os.path.exists(result_html_path):
    with open(result_html_path, "w") as f:
        f.write("""
<!DOCTYPE html>
<html>
<head>
    <title>Detection Results</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2 {
            color: #333;
        }
        .result-container {
            margin-top: 30px;
        }
        .video-container {
            margin: 20px 0;
        }
        video {
            max-width: 100%;
        }
        .stats {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin-top: 20px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .alert-sent {
            background-color: #d4edda;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
        .alert-not-sent {
            background-color: #f8d7da;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <h1>Detection Results</h1>
    
    <div class="stats">
        <h2>Prediction: {{ predicted_class }}</h2>
        <p><strong>Confidence:</strong> {{ "%.2f"|format(confidence*100) }}%</p>
        <p><strong>Processing Time:</strong> {{ "%.2f"|format(processing_time) }} seconds</p>
        
        <!-- Always show WhatsApp alert status -->
        {% if alert_sent %}
        <div class="alert-sent">
            <p><strong>WhatsApp Alert Sent!</strong> A notification was sent to {{ phone_number }}.</p>
        </div>
        {% else %}
        <div class="alert-not-sent">
            <p><strong>No Alert Sent:</strong> {{ alert_message }}</p>
        </div>
        {% endif %}
    </div>
    
    <div class="result-container">
        <h2>Original Video</h2>
        <div class="video-container">
            <video controls>
                <source src="{{ url_for('static', filename='uploads/' + original_video) }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <h2>Visualization with Predictions</h2>
        <div class="video-container">
            <video controls>
                <source src="{{ url_for('static', filename='uploads/' + prediction_video) }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <h2>Prediction Distribution</h2>
        <table>
            <tr>
                <th>Class</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
            {% for class_name, data in class_counts.items() %}
            <tr>
                <td>{{ class_name }}</td>
                <td>{{ data.count }}</td>
                <td>{{ "%.2f"|format(data.percentage) }}%</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <a href="{{ url_for('index') }}" class="btn">Analyze Another Video</a>
</body>
</html>
""")

app = Flask(__name__, 
           template_folder=template_dir,
           static_folder=static_dir)

app.config['SECRET_KEY'] = 'ucf_crime_detection_app'
app.config['UPLOAD_FOLDER'] = os.path.join(static_dir, 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
app.config['MODEL_PATH'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models', 'best_model.pth')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max upload size

# Twilio configuration - made optional
app.config['TWILIO_ACCOUNT_SID'] = os.environ.get('TWILIO_ACCOUNT_SID', '')
app.config['TWILIO_AUTH_TOKEN'] = os.environ.get('TWILIO_AUTH_TOKEN', '')
app.config['TWILIO_FROM_NUMBER'] = os.environ.get('TWILIO_FROM_NUMBER', '')  
app.config['FIXED_RECEIVER_NUMBER'] = '+919080918203'  # Fixed receiver number

# Gemini configuration - optional; falls back to template text
app.config['GOOGLE_API_KEY'] = os.environ.get('GOOGLE_API_KEY', '')
app.config['GEMINI_MODEL'] = os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash')

# Optional public base URL to serve media to Twilio (must be public HTTPS)
# Example: https://your-domain.example.com/static/uploads
app.config['MEDIA_BASE_URL'] = os.environ.get('MEDIA_BASE_URL', '')

# Supabase configuration (optional, enables permanent public URLs without a domain)
app.config['SUPABASE_URL'] = os.environ.get('SUPABASE_URL', '')
app.config['SUPABASE_SERVICE_KEY'] = os.environ.get('SUPABASE_SERVICE_KEY', '')
app.config['SUPABASE_BUCKET'] = os.environ.get('SUPABASE_BUCKET', 'media')
try:
    from supabase import create_client as _create_supabase_client  # type: ignore
    SUPABASE_AVAILABLE = True
except Exception:
    _create_supabase_client = None
    SUPABASE_AVAILABLE = False

# If env was not provided, fall back to user-provided values (local default)
if not app.config['SUPABASE_URL']:
    app.config['SUPABASE_URL'] = 'https://joronooxhbccffvbshry.supabase.co'
if not app.config['SUPABASE_SERVICE_KEY']:
    app.config['SUPABASE_SERVICE_KEY'] = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Impvcm9ub294aGJjY2ZmdmJzaHJ5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTAyMzUzMywiZXhwIjoyMDc0NTk5NTMzfQ.TFzGHntsg4Uj9FLJDRhmUUGAkcQ9bWpL_ZAmBemy5XQ'

# Check if WhatsApp/Twilio is properly configured
WHATSAPP_ENABLED = all([
    app.config['TWILIO_ACCOUNT_SID'],
    app.config['TWILIO_AUTH_TOKEN'],
    app.config['TWILIO_FROM_NUMBER']
])

# Ensure upload and output directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Class names
CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

# High severity classes that should trigger alerts by default
DEFAULT_ALERT_CLASSES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

# Load model on startup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Custom wrapper for torch.load to handle warnings
def load_model_safe(model_path, device):
    try:
        # First try with weights_only=True (PyTorch newer versions)
        return torch.load(model_path, map_location=device, weights_only=True)
    except (TypeError, Exception) as e:
        # If that fails, fall back to the original method (trusted source)
        print(f"Weights-only loading failed, using legacy loading for trusted model: {str(e)[:100]}...")
        return torch.load(model_path, map_location=device, weights_only=False)

# Initialize the model
def load_detection_model():
    global model
    
    if not MODEL_AVAILABLE:
        print("No model architectures available. Please check model imports.")
        return None
    
    try:
        print(f"Attempting to load model from {app.config['MODEL_PATH']}")
        
        # Use the custom wrapper for torch.load
        checkpoint = load_model_safe(app.config['MODEL_PATH'], device)
        
        # Check the structure of the checkpoint
        print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dictionary'}")
        
        model = None
        
        # Try to load with MaxAccuracyC3D first (matches the actual checkpoint)
        if MAXACCURACY_MODEL_AVAILABLE:
            try:
                print("Attempting to load with MaxAccuracyC3D...")
                # Use the same configuration as found in the checkpoint
                model = MaxAccuracyC3D(
                    num_classes=14, 
                    dropout_rate=0.3, 
                    use_attention=True
                )
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        # Assume the whole dict is the state dict
                        model.load_state_dict(checkpoint)
                else:
                    model = checkpoint
                
                print("MaxAccuracyC3D loaded successfully")
            except Exception as e:
                print(f"Failed to load with MaxAccuracyC3D: {str(e)}")
                import traceback
                traceback.print_exc()
                model = None
        
        # Fall back to C3D if AdvancedVideoModel failed
        if model is None and C3D_AVAILABLE:
            try:
                print("Attempting to load with C3D...")
                model = C3D(num_classes=14)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model.load_state_dict(checkpoint)
                else:
                    model = checkpoint
                
                print("C3D loaded successfully")
            except Exception as e:
                print(f"Failed to load with C3D: {str(e)}")
                model = None
        
        if model is None:
            print("All model loading attempts failed")
            return None
        
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        
        # Test the model with a small dummy input
        dummy_input = torch.zeros((1, 3, 16, 112, 112), device=device)
        with torch.no_grad():
            _ = model(dummy_input)
        print("Model successfully processed a test input")
        
        return model
        
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# Generate a timestamped event-based filename
def generate_event_filename(original_filename, event_name, prefix=""):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    extension = original_filename.rsplit('.', 1)[1].lower()
    sanitized_event = event_name.replace(' ', '_').lower()
    return f"{prefix}{sanitized_event}_{timestamp}.{extension}"

# --- Supabase helpers ---
def get_supabase_client():
    if not (SUPABASE_AVAILABLE and app.config['SUPABASE_URL'] and app.config['SUPABASE_SERVICE_KEY']):
        return None
    try:
        return _create_supabase_client(app.config['SUPABASE_URL'], app.config['SUPABASE_SERVICE_KEY'])
    except Exception:
        return None

def upload_media_to_supabase(local_path: str, dest_name: str) -> str:
    """Upload a local file to Supabase public bucket using REST API to ensure correct content-type.
    Returns public URL or empty string on failure."""
    client = get_supabase_client()
    if client is None:
        print("[Supabase] Client not available")
        return ''
    bucket = app.config['SUPABASE_BUCKET'] or 'media'
    supabase_url = app.config['SUPABASE_URL']
    service_key = app.config['SUPABASE_SERVICE_KEY']
    
    # Determine content type with explicit overrides
    ext = os.path.splitext(dest_name)[1].lower()
    if ext in {'.mp4', '.m4v'}:
        content_type = 'video/mp4'
    elif ext in {'.jpg', '.jpeg'}:
        content_type = 'image/jpeg'
    elif ext in {'.png'}:
        content_type = 'image/png'
    else:
        content_type, _ = mimetypes.guess_type(dest_name)
        if not content_type:
            content_type = 'application/octet-stream'
    
    # Use a date-based path for uniqueness
    subdir = time.strftime('%Y/%m/%d')
    object_path = f"{subdir}/{dest_name}"
    
    try:
        # Use REST API directly to ensure content-type header is set correctly
        upload_url = f"{supabase_url}/storage/v1/object/{bucket}/{object_path}"
        headers = {
            'Authorization': f'Bearer {service_key}',
            'Content-Type': content_type,
            'x-upsert': 'true'
        }
        
        with open(local_path, 'rb') as f:
            file_data = f.read()
        
        print(f"[Supabase] Uploading {dest_name} ({len(file_data)} bytes) as {content_type}")
        response = requests.put(upload_url, headers=headers, data=file_data, timeout=30)
        
        if response.status_code not in (200, 201):
            print(f"[Supabase] Upload failed: {response.status_code} - {response.text[:200]}")
            return ''
        
        # Get public URL
        public_url = f"{supabase_url}/storage/v1/object/public/{bucket}/{object_path}"
        
        # Verify the content-type is correct
        try:
            verify_resp = requests.head(public_url, timeout=10)
            if verify_resp.status_code == 405:
                verify_resp = requests.get(public_url, stream=True, timeout=10)
            actual_ctype = verify_resp.headers.get('Content-Type', '')
            print(f"[Supabase] Uploaded {dest_name}, public URL content-type: {actual_ctype}")
            if not (actual_ctype.startswith('image/') or actual_ctype.startswith('video/')):
                print(f"[Supabase] WARNING: Content-type mismatch! Expected media type, got {actual_ctype}")
        except Exception as e:
            print(f"[Supabase] Could not verify content-type: {e}")
        
        return public_url
    except Exception as e:
        print(f"[Supabase] Upload error: {e}")
        import traceback
        traceback.print_exc()
        return ''

def build_public_urls_for_media(local_filenames: List[str]) -> List[str]:
    """Turn local filenames in static/uploads into internet-reachable URLs.
    Prefers Supabase if configured; falls back to MEDIA_BASE_URL.
    """
    public_urls: List[str] = []
    uploads_dir = app.config['UPLOAD_FOLDER']
    # Prefer Supabase
    if get_supabase_client() is not None:
        for name in local_filenames:
            local_path = os.path.join(uploads_dir, name)
            url = upload_media_to_supabase(local_path, name)
            if url:
                public_urls.append(url)
    # Fallback to MEDIA_BASE_URL
    if not public_urls and app.config['MEDIA_BASE_URL']:
        base = app.config['MEDIA_BASE_URL'].rstrip('/')
        for name in local_filenames:
            public_urls.append(f"{base}/{name}")
    return public_urls

def validate_media_urls(urls: List[str]) -> List[str]:
    """Filter URLs to those that are publicly reachable with a valid media content-type."""
    valid: List[str] = []
    for u in urls:
        try:
            print(f"[Validation] Checking {u}")
            r = requests.head(u, timeout=10, allow_redirects=True)
            if r.status_code == 405:  # some CDNs disallow HEAD
                r = requests.get(u, stream=True, timeout=10, allow_redirects=True)
            ctype = r.headers.get('Content-Type', '')
            print(f"[Validation] Status: {r.status_code}, Content-Type: {ctype}")
            if r.status_code == 200 and (
                ctype.startswith('image/') or ctype.startswith('video/')
            ):
                print(f"[Validation] ✓ Valid media URL: {u}")
                valid.append(u)
            else:
                print(f"[Validation] ✗ Invalid: status={r.status_code}, type={ctype}")
        except Exception as e:
            print(f"[Validation] ✗ Error checking {u}: {e}")
            continue
    return valid

# --- Gemini helper: generate dynamic alert text ---
def _normalize_gemini_model(name: str) -> str:
    """Return a stable, supported Gemini model string for REST (prefer -latest)."""
    if not name:
        return 'gemini-2.0-flash'
    name = name.strip()
    # Map common aliases to stable names (prefer current GA models)
    if name.startswith('gemini-1.5'):
        # Migrate 1.5 aliases to a widely-available GA model
        return 'gemini-2.0-flash'
    if name in ('gemini-1.5-flash', 'gemini-1.5-flash-latest'):
        return 'gemini-2.0-flash'
    if name in ('gemini-1.5-pro', 'gemini-1.5-pro-latest'):
        return 'gemini-2.0-flash'
    if name in ('gemini-1.5-flash-8b', 'gemini-1.5-flash-8b-latest'):
        return 'gemini-2.0-flash'
    if name in ('gemini-2.0-flash', 'gemini-2.0-flash-exp'):
        return 'gemini-2.0-flash'
    # Default: append -latest for better availability
    return name

def generate_gemini_alert_text(event_name: str, confidence_pct: float, duration_s: float,
                               extra_context: Optional[Dict[str, str]] = None,
                               snapshot_filenames: Optional[List[str]] = None) -> tuple[str, str]:
    """Generate a concise human-friendly alert using Gemini if configured; otherwise a fallback string."""
    # Fallback text (no external dependency)
    fallback = (
        f"Incident: Likely {event_name} detected (confidence {confidence_pct:.1f}%).\n"
        f"Why: Visual cues consistent with {event_name.lower()} observed in recent frames.\n"
        "Actions:\n- Verify on live feed.\n- Alert on-site staff if risk persists.\n- Preserve footage for review."
    )
    if not (GEMINI_AVAILABLE and app.config['GOOGLE_API_KEY']):
        return fallback
    try:
        genai.configure(api_key=app.config['GOOGLE_API_KEY'])
        model_name = _normalize_gemini_model(app.config['GEMINI_MODEL'])
        model = genai.GenerativeModel(model_name)
        context_lines = []
        if extra_context:
            for k, v in extra_context.items():
                context_lines.append(f"{k}: {v}")
        ctx = "\n".join(context_lines)
        prompt_text = (
            "Role: on-call security operator assistant.\n"
            "Goal: draft a professional WhatsApp alert (plain text, no markdown/emojis).\n"
            "Style: concise, human, operational; avoid AI phrasing and hedging.\n"
            "Format exactly:\n"
            "[ALERT] <AnomalyType> — short scene description (what/where).\n"
            "Why: key visual cues supporting the call.\n"
            "Actions: two short, practical steps separated by '; '.\n"
            "Rules: keep under 280 chars; do not list 'confidence' unless >= 60%; no personal identifiers or speculation.\n"
            f"AnomalyType: {event_name}\n"
            f"ConfidencePercent: {confidence_pct:.1f}\n"
            f"ClipDurationSec: {duration_s:.1f}\n"
            f"{ctx}\n"
        )

        # Build content parts, optionally including snapshots as images
        content_parts = [prompt_text]
        if snapshot_filenames:
            # Attach up to two images as inline parts
            for fname in snapshot_filenames[:2]:
                img_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                try:
                    with open(img_path, 'rb') as f:
                        content_parts.append({
                            'mime_type': 'image/jpeg',
                            'data': f.read()
                        })
                except Exception:
                    continue

        try:
            resp = model.generate_content(content_parts)
        except Exception:
            # Try fast fallback model
            model = genai.GenerativeModel('gemini-1.5-flash')
            resp = model.generate_content(content_parts)
        text = (resp.text or "").strip()
        # Basic guardrails
        if not text:
            return fallback, 'fallback'
        if len(text) > 400:
            text = text[:397] + "..."
        return text, 'gemini'
    except Exception:
        # REST fallback for broader compatibility on older Python
        try:
            api_key = app.config['GOOGLE_API_KEY']
            model_candidates = [_normalize_gemini_model(app.config['GEMINI_MODEL'])]
            # Single stable API version preferred by docs
            versions = ["v1beta"]

            parts = [{"text": prompt_text}]
            if snapshot_filenames:
                for fname in snapshot_filenames[:2]:
                    img_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
                    try:
                        with open(img_path, 'rb') as f:
                            b64 = base64.b64encode(f.read()).decode('utf-8')
                            parts.append({
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64
                                }
                            })
                    except Exception:
                        continue

            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 120,
                    "topP": 0.9
                }
            }
            headers = {"Content-Type": "application/json"}

            for ver in versions:
                for name in model_candidates:
                    url = f"https://generativelanguage.googleapis.com/{ver}/models/{name}:generateContent?key={api_key}"
                    try:
                        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
                        print(f"[Gemini-REST] try {ver} {name} -> {r.status_code}")
                        if r.status_code == 200:
                            data = r.json()
                            text = ""
                            try:
                                cands = data.get("candidates", [])
                                if cands:
                                    parts_out = cands[0].get("content", {}).get("parts", [])
                                    txts = [p.get("text", "") for p in parts_out if "text" in p]
                                    text = "\n".join(t for t in txts if t).strip()
                            except Exception:
                                text = ""
                            if text:
                                if len(text) > 400:
                                    text = text[:397] + "..."
                                return text, 'gemini-rest'
                    except Exception:
                        continue
            return fallback, 'fallback'
        except Exception:
            return fallback, 'fallback'

# --- Video helper: save 1-2 anomaly frames near peak-confidence clip ---
def save_anomaly_frames(video_path: str, best_clip_idx: int, event_name: str,
                        clip_len: int = 16, sample_rate: int = 10, overlap: float = 0.5,
                        count: int = 2) -> List[str]:
    """Extract representative frames near the peak-confidence clip and save into uploads. Returns filenames."""
    try:
        import cv2
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        # Map clip index to approximate source frame index
        step = max(1, int(clip_len * (1 - overlap)))
        sampled_start = best_clip_idx * step
        # Choose center and an offset frame within the clip window
        sampled_centers = [sampled_start + clip_len // 2]
        if count > 1:
            sampled_centers.append(sampled_start + min(clip_len - 1, (3 * clip_len) // 4))
        # Convert sampled indices to original frame numbers
        frame_indices = [max(0, idx * sample_rate) for idx in sampled_centers]
        cap = cv2.VideoCapture(video_path)
        saved_files = []
        for i, frame_idx in enumerate(frame_indices[:count]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                continue
            fname = generate_event_filename("frame.jpg", event_name, prefix=f"snapshot{i+1}_")
            out_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            cv2.imwrite(out_path, frame)
            saved_files.append(fname)
        cap.release()
        return saved_files
    except Exception:
        return []

# Function to send WhatsApp alert with Twilio
def send_whatsapp_alert(to_number, event_name, confidence, location="Unknown Location", *,
                        message_override: Optional[str] = None,
                        media_filenames: Optional[List[str]] = None,
                        media_urls: Optional[List[str]] = None):
    # Check if WhatsApp/Twilio is properly configured
    if not WHATSAPP_ENABLED:
        print("WhatsApp alerts are not configured. Set TWILIO environment variables to enable.")
        return False, "WhatsApp alerts not configured. Please set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER environment variables."
    
    try:
        # Always use the fixed number regardless of input
        to_number = app.config['FIXED_RECEIVER_NUMBER']

        # Normalize Twilio numbers for WhatsApp
        from_number = app.config['TWILIO_FROM_NUMBER'].strip()
        if not from_number:
            return False, "TWILIO_FROM_NUMBER is empty"
        # Ensure from_number has whatsapp: prefix
        if not from_number.startswith('whatsapp:'):
            # Accept numbers with or without leading +
            if from_number.startswith('+'):
                from_number = f"whatsapp:{from_number}"
            else:
                from_number = f"whatsapp:+{from_number}"

        # Ensure to_number starts with whatsapp:+<countrycode><number>
        normalized_to = to_number.strip()
        if not normalized_to.startswith('+'):
            normalized_to = f"+{normalized_to}"
        whatsapp_to = f"whatsapp:{normalized_to}"

        # Initialize Twilio client with your credentials
        client = Client(app.config['TWILIO_ACCOUNT_SID'], app.config['TWILIO_AUTH_TOKEN'])
        
        # Build message body (Gemini override if provided)
        if message_override:
            message_body = message_override.strip()
            print(f"[WhatsApp] Using AI-generated (Gemini) alert message: {message_body[:80]}...")
        else:
            message_body = f"⚠️ ALERT: {event_name} detected with {confidence:.1f}% confidence.\n"
            message_body += f"Location: {location}\n"
            message_body += f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            message_body += "Please check the surveillance system immediately."
            print(f"[WhatsApp] Using default (non-AI) alert message")
        
        # When media is attached, ensure message is concise (Twilio WhatsApp limit ~1600 chars with media)
        # Keep essential info but truncate if too long
        if media_urls or media_filenames:
            if len(message_body) > 1200:
                # Keep first part and truncate
                message_body = message_body[:1200].rsplit('\n', 1)[0] + "\n[View attached media for details]"

        # Optional media (requires MEDIA_BASE_URL to be a public HTTPS host)
        # Determine media URLs
        if media_urls is None and media_filenames:
            try:
                media_urls = build_public_urls_for_media(media_filenames)
                print(f"[WhatsApp] Built {len(media_urls)} media URLs from {len(media_filenames)} files")
            except Exception as e:
                print(f"[WhatsApp] Failed to build media URLs: {e}")
                media_urls = None

        # Validate and send media
        valid_media = []
        if media_urls:
            valid_media = validate_media_urls(media_urls)
            print(f"[WhatsApp] Validated {len(valid_media)}/{len(media_urls)} media URLs")
            if valid_media:
                print(f"[WhatsApp] Sending media: {valid_media[0]}")
            else:
                print(f"[WhatsApp] WARNING: No valid media URLs! All {len(media_urls)} URLs failed validation")

        # Ensure message body is not empty (required by Twilio even with media)
        if not message_body or not message_body.strip():
            message_body = f"⚠️ ALERT: {event_name} detected with {confidence:.1f}% confidence."
        
        # Log what we're about to send
        print(f"[WhatsApp] Message body preview: {message_body[:100]}...")
        print(f"[WhatsApp] Message body full length: {len(message_body)} chars")

        # Send the WhatsApp message (with or without media)
        if valid_media:
            # Twilio WhatsApp sometimes drops body when media_url is included
            # Solution: Send text message first, then media with caption
            media_url_list = valid_media[:1] if isinstance(valid_media, list) else [valid_media[0]]
            
            # Step 1: Send the full AI-generated alert text first
            print(f"[WhatsApp] Step 1: Sending text message first. Body: '{message_body[:50]}...'")
            try:
                text_message = client.messages.create(
                    body=message_body,
                    from_=from_number,
                    to=whatsapp_to
                )
                print(f"[WhatsApp] Text message sent. SID: {text_message.sid}")
            except Exception as e:
                print(f"[WhatsApp] ERROR sending text message: {e}")
                text_message = None
            
            # Step 2: Send media with a short caption (optional, but helps if first message fails)
            print(f"[WhatsApp] Step 2: Sending media with caption. Media: {media_url_list[0]}")
            try:
                # Short caption summarizing the alert
                media_caption = f"📹 {event_name} detected ({confidence:.1f}% confidence)"
                message = client.messages.create(
                    body=media_caption,  # Short caption for media
                    from_=from_number,
                    to=whatsapp_to,
                    media_url=media_url_list
                )
                print(f"[WhatsApp] Media message sent. SID: {message.sid}")
                # Return the text message SID as primary (contains the full AI message)
                message = text_message if text_message else message
            except Exception as e:
                print(f"[WhatsApp] ERROR sending media: {e}")
                # If media fails, we at least sent the text
                message = text_message if text_message else None
                if not message:
                    raise Exception("Failed to send both text and media")
        else:
            print(f"[WhatsApp] Sending message WITHOUT media. Body: '{message_body[:50]}...'")
            message = client.messages.create(
                body=message_body,
                from_=from_number,
                to=whatsapp_to
            )
        
        print(f"WhatsApp alert sent to {whatsapp_to}: {message.sid}")
        return True, message.sid
    except Exception as e:
        print(f"Failed to send WhatsApp alert: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, str(e)

@app.route('/')
def index():
    global model
    # Initialize the model if it's not loaded yet
    if model is None:
        model = load_detection_model()
    return render_template('index.html', class_names=CLASS_NAMES)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global model
    
    # Handle GET requests to /upload by redirecting to the index page
    if request.method == 'GET':
        return redirect(url_for('index'))
    
    # Check if model is loaded
    if model is None:
        model = load_detection_model()  # Try to load model
        if model is None:  # If still None, show error
            flash('Model not loaded. Please try again later.', 'error')
            return redirect(url_for('index'))
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        flash('No video selected', 'error')
        return redirect(url_for('index'))
    
    # Get WhatsApp alert settings - alerts are always enabled with fixed number
    alert_classes = request.form.getlist('alert_classes')
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Use fixed number
    
    # If no classes selected, use defaults
    if not alert_classes:
        alert_classes = DEFAULT_ALERT_CLASSES
    
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        # Use temporary UUID for processing
        temp_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        
        # Save the file
        file.save(filepath)
        
        # Redirect to processing page with alert settings as query parameters
        return redirect(url_for('process_video', 
                               filename=temp_filename, 
                               original_filename=file.filename,
                               alert_classes=','.join(alert_classes),
                               phone_number=phone_number))
    
    flash('File type not allowed. Please upload mp4, avi, mov, or mkv.', 'error')
    return redirect(url_for('index'))

@app.route('/process/<filename>')
def process_video(filename):
    global model
    # Get original filename from query params if available
    original_filename = request.args.get('original_filename', filename)
    
    # Get alert settings from query parameters - use fixed number
    alert_classes = request.args.get('alert_classes', '').split(',') if request.args.get('alert_classes') else DEFAULT_ALERT_CLASSES
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Always use fixed number
    
    # Ensure model is loaded
    if model is None:
        model = load_detection_model()
        if model is None:
            flash('Model not loaded. Please try again later.', 'error')
            return redirect(url_for('index'))
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    # Check if file exists
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('index'))
    
    try:
        # Process the video
        start_time = time.time()
        pred_class, confidence, all_preds, all_confs = predict_video(
            model, filepath, clip_len=16, device=device
        )
        processing_time = time.time() - start_time
        
        if pred_class is not None:
            # Create prediction distribution
            class_counts = {}
            for i, count in enumerate(torch.bincount(torch.tensor(all_preds), minlength=len(CLASS_NAMES))):
                if count > 0:
                    class_counts[CLASS_NAMES[i]] = {
                        'count': count.item(),
                        'percentage': count.item() / len(all_preds) * 100
                    }
            
            # Get the predicted event name
            event_name = CLASS_NAMES[pred_class]
            
            # Generate event-based filenames
            event_original_filename = generate_event_filename(original_filename, event_name, "original_")
            event_result_filename = generate_event_filename(original_filename, event_name, "result_")
            
            # Copy original file with event name
            original_event_path = os.path.join(app.config['OUTPUT_FOLDER'], event_original_filename)
            with open(filepath, 'rb') as src_file, open(original_event_path, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            # Also keep a copy in uploads folder for web display
            display_original = os.path.join(app.config['UPLOAD_FOLDER'], event_original_filename)
            with open(filepath, 'rb') as src_file, open(display_original, 'wb') as dst_file:
                dst_file.write(src_file.read())
            
            # Create visualization and save with event-based name
            vis_filepath_web = os.path.join(app.config['UPLOAD_FOLDER'], event_result_filename)
            vis_filepath_output = os.path.join(app.config['OUTPUT_FOLDER'], event_result_filename)
            
            # Create output directory for visualization if it doesn't exist
            os.makedirs(os.path.dirname(vis_filepath_output), exist_ok=True)
            
            # Save the visualization
            vis_path = save_prediction_visualization(
                filepath, all_preds, all_confs, CLASS_NAMES,
                output_path=vis_filepath_web
            )
            
            # Also save a copy to the output folder (use actual path returned in case extension changed)
            actual_vis_web = vis_path if vis_path else vis_filepath_web
            if os.path.exists(actual_vis_web):
                with open(actual_vis_web, 'rb') as src_file, open(vis_filepath_output, 'wb') as dst_file:
                    dst_file.write(src_file.read())
            
            # Prepare Gemini alert text and anomaly snapshots
            # Determine best clip index for snapshots
            best_clip_idx = None
            try:
                if all_preds and all_confs:
                    best = -1.0
                    idx = None
                    for i, (p, c) in enumerate(zip(all_preds, all_confs)):
                        if p == pred_class and c > best:
                            best = c
                            idx = i
                    best_clip_idx = idx
            except Exception:
                best_clip_idx = None

            # Save 1-2 representative frames
            snapshot_files = []
            if best_clip_idx is not None:
                snapshot_files = save_anomaly_frames(
                    filepath, best_clip_idx, event_name,
                    clip_len=16, sample_rate=10, overlap=0.5, count=2
                )

            # Try to export a short anomaly clip (3s) for WhatsApp media
            clip_file = ''
            try:
                # Map best_clip_idx to approximate center frame index in original video
                step = max(1, int(16 * (1 - 0.5)))  # consistent with predict_video
                sampled_center = (best_clip_idx if best_clip_idx is not None else 0) * step + 8
                center_frame = sampled_center * 10  # sample_rate=10 in predict_video
                clip_file = export_anomaly_clip(
                    filepath,
                    center_frame=center_frame,
                    seconds=3.0,
                    out_dir=app.config['UPLOAD_FOLDER'],
                    prefix=f"clip_{event_name.lower()}_"
                )
            except Exception:
                clip_file = ''

            # Generate dynamic alert text (Gemini if available)
            alert_text, alert_source = generate_gemini_alert_text(
                event_name=event_name,
                confidence_pct=confidence * 100.0,
                duration_s=processing_time,
                extra_context={
                    "TopPrediction": event_name,
                    "Confidence": f"{confidence*100:.1f}%",
                },
                snapshot_filenames=snapshot_files
            )

            # Send WhatsApp alert if class is in alert list
            alert_sent = False
            alert_message = "The detected event is not in your alert list."
            
            if event_name in alert_classes:
                media_list = []
                if clip_file:
                    media_list.append(clip_file)
                elif snapshot_files:
                    media_list.extend(snapshot_files)

                # Turn local filenames into public URLs (Supabase preferred)
                public_urls = build_public_urls_for_media(media_list) if media_list else None

                success, message_id = send_whatsapp_alert(
                    phone_number,
                    event_name,
                    confidence * 100,
                    message_override=alert_text,
                    media_filenames=media_list,
                    media_urls=public_urls
                )
                alert_sent = success
                if not success:
                    alert_message = f"Error sending alert: {message_id}"
            
            # Pass just the filenames, not the relative paths
            return render_template(
                'result.html',
                original_video=event_original_filename,
                prediction_video=event_result_filename,
                predicted_class=event_name,
                confidence=confidence,
                processing_time=processing_time,
                class_counts=class_counts,
                alert_sent=alert_sent,
                alert_message=alert_message,
                phone_number=phone_number,
                alert_text=alert_text,
                alert_source=alert_source,
                snapshots=snapshot_files
            )
        else:
            flash('Could not process video. Please try another one.', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error processing video: {str(e)}', 'error')
        import traceback
        traceback.print_exc()
        return redirect(url_for('index'))
    finally:
        # Cleanup temporary file if needed
        if os.path.exists(filepath) and filename.startswith(str(uuid.UUID(int=0))[0:8]):
            try:
                os.remove(filepath)
            except:
                pass

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic video prediction"""
    global model
    # Check if model is loaded
    if model is None:
        model = load_detection_model()  # Try to load model
        if model is None:  # If still None, return error
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
    
    # Check if a file was uploaded
    if 'video' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No video file provided'
        }), 400
    
    file = request.files['video']
    
    # Check if file was selected
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    # Get alert settings from form or JSON data - use fixed number
    data = request.form.to_dict() if request.form else request.get_json(silent=True) or {}
    alert_classes = data.get('alert_classes', '').split(',') if isinstance(data.get('alert_classes'), str) else data.get('alert_classes', [])
    phone_number = app.config['FIXED_RECEIVER_NUMBER']  # Always use fixed number
    
    # If no classes selected, use defaults
    if not alert_classes:
        alert_classes = DEFAULT_ALERT_CLASSES
    
    # Check if file type is allowed
    if file and allowed_file(file.filename):
        temp_filename = str(uuid.uuid4()) + '.' + file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(filepath)
        
        try:
            # Process the video
            pred_class, confidence, all_preds, all_confs = predict_video(
                model, filepath, clip_len=16, device=device
            )
            
            if pred_class is not None:
                # Create prediction distribution
                class_distribution = {}
                for i, count in enumerate(torch.bincount(torch.tensor(all_preds), minlength=len(CLASS_NAMES))):
                    if count > 0:
                        class_distribution[CLASS_NAMES[i]] = {
                            'count': count.item(),
                            'percentage': count.item() / len(all_preds) * 100
                        }
                
                # Get the predicted event name
                event_name = CLASS_NAMES[pred_class]
                
                # Generate event-based filenames
                event_original_filename = generate_event_filename(file.filename, event_name, "original_")
                event_result_filename = generate_event_filename(file.filename, event_name, "result_")
                
                # Save files to output directory with event-based names
                original_event_path = os.path.join(app.config['OUTPUT_FOLDER'], event_original_filename)
                with open(filepath, 'rb') as src_file, open(original_event_path, 'wb') as dst_file:
                    dst_file.write(src_file.read())
                
                # Create visualization and save with event-based name
                vis_filepath = os.path.join(app.config['OUTPUT_FOLDER'], event_result_filename)
                
                # Create output directory for visualization if it doesn't exist
                os.makedirs(os.path.dirname(vis_filepath), exist_ok=True)
                
                # Save the visualization
                vis_path = save_prediction_visualization(
                    filepath, all_preds, all_confs, CLASS_NAMES,
                    output_path=vis_filepath
                )
                
                # Send WhatsApp alert if class is in alert list
                alert_sent = False
                alert_message = "The detected event is not in the alert list."
                alert_message_id = None
                
                if event_name in alert_classes:
                    success, message_id = send_whatsapp_alert(
                        phone_number,
                        event_name,
                        confidence * 100
                    )
                    alert_sent = success
                    alert_message_id = message_id
                    if not success:
                        alert_message = f"Error sending alert: {message_id}"
                
                # Create response
                response = {
                    'success': True,
                    'prediction': {
                        'class': event_name,
                        'class_id': int(pred_class),
                        'confidence': float(confidence),
                        'distribution': class_distribution
                    },
                    'files': {
                        'original': event_original_filename,
                        'result': event_result_filename,
                        'path': app.config['OUTPUT_FOLDER']
                    },
                    'alert': {
                        'sent': alert_sent,
                        'message': alert_message if not alert_sent else "Alert sent successfully",
                        'message_id': alert_message_id,
                        'phone': phone_number
                    }
                }
                
                return jsonify(response), 200
            else:
                return jsonify({
                    'success': False,
                    'error': 'Could not process video'
                }), 500
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
        finally:
            # Clean up the temporary file
            try:
                os.remove(filepath)
            except:
                pass
    
    return jsonify({
        'success': False,
        'error': 'File type not allowed'
    }), 400

# Add a special route to serve files from the static/uploads directory
@app.route('/static/uploads/<filename>')
def serve_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    # Load the model at startup
    model = load_detection_model()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)