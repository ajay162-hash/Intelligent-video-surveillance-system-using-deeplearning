# Intelligent Video Surveillance System Using Deep Learning

An advanced video surveillance system that uses deep learning to detect and classify various types of criminal activities in real-time. The system provides WhatsApp alerts for critical security events.

## Features

- **Real-time Crime Detection**: Detects 14 different types of criminal activities
- **WhatsApp Alerts**: Automatic notifications for high-priority security events
- **Web Interface**: Easy-to-use web application for video upload and analysis
- **Event-based Logging**: Automatically saves processed videos with timestamps
- **API Support**: REST API for programmatic access

## Detected Crime Types

The system can detect the following types of activities:
- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Normal (safe activities)
- Road Accidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism

## Dataset

This project uses the **UCF Crime Dataset**, which contains 14 different types of criminal activities and normal activities:

### Automatic Dataset Setup

Use the provided script to automatically download and preprocess the dataset:

```bash
python download_dataset.py
```

This script will:
- Download the UCF Crime dataset videos
- Extract frames from videos (PNG format)
- Organize data into train/test folders by class
- Set up the correct directory structure

### Manual Dataset Setup

If you prefer to set up the dataset manually:

1. **Download UCF Crime Dataset**:
   - Visit: http://crcv.ucf.edu/projects/real-world/
   - Download the "Anomaly Detection in Videos" dataset
   - Extract all video files

2. **Extract frames from videos**:
   ```bash
   python preprocess_data.py --input_dir /path/to/videos --output_dir /path/to/frames
   ```

3. **Organize the data structure**:
   ```
   data/
   ├── train/
   │   ├── Abuse/
   │   │   ├── frame_001.png
   │   │   ├── frame_002.png
   │   │   └── ...
   │   ├── Arrest/
   │   ├── Arson/
   │   └── ... (other classes)
   └── test/
       ├── Abuse/
       ├── Arrest/
       └── ... (other classes)
   ```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ajay162-hash/Intelligent-video-surveillance-system-using-deeplearning.git
   cd Intelligent-video-surveillance-system-using-deeplearning
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up the dataset** (choose one option):
   
   **Option A - Automatic (Recommended):**
   ```bash
   python download_dataset.py
   ```
   
   **Option B - Manual:**
   - Download UCF Crime dataset manually
   - Run: `python preprocess_data.py --input_dir /path/to/videos --output_dir data`

4. **Train the model**:
   ```bash
   python train.py --train_dir data/train --test_dir data/test --epochs 50
   ```
   
   *Note: The trained model file is not included in the repository due to size constraints. You must train your own model before using the web application.*

5. **Set up environment variables** (optional, for WhatsApp alerts):
   ```bash
   export TWILIO_ACCOUNT_SID="your_twilio_account_sid"
   export TWILIO_AUTH_TOKEN="your_twilio_auth_token"
   export TWILIO_FROM_NUMBER="your_twilio_phone_number"
   ```

## Usage

### Web Application

1. **Start the server**:
   ```bash
   python app.py
   ```

2. **Open your browser** and go to `http://localhost:5000`

3. **Upload a video** and get real-time analysis results

### Demo Videos

To test the system immediately without real surveillance footage:

1. **Generate demo videos**:
   ```bash
   python create_demo_videos.py
   ```

2. **Use the demo videos**: Upload any video from the `demo_videos/` folder to test different crime detection scenarios

### Training Your Own Model

1. **Prepare your dataset**:
   - Organize your data in folders by class name
   - Each folder should contain PNG frames extracted from videos

2. **Train the model**:
   ```bash
   python train.py --train_dir /path/to/train/data --test_dir /path/to/test/data --epochs 50
   ```

3. **Monitor training**:
   - Training logs and checkpoints are saved in `trained_models/`
   - Best model is automatically saved as `trained_models/best_model.pth`

### API Usage

Send a POST request to `/api/predict` with a video file:

```python
import requests

files = {'video': open('your_video.mp4', 'rb')}
response = requests.post('http://localhost:5000/api/predict', files=files)
result = response.json()
```

## Project Structure

```
organized_project/
├── app.py                 # Main Flask web application
├── train.py              # Training script
├── models/               # Model architectures
│   ├── __init__.py
│   ├── advanced_model.py # Advanced C3D with attention
│   └── cnn3d.py         # Standard C3D model
├── utils/               # Utility functions
│   └── video_utils.py   # Video processing utilities
├── trained_models/      # Saved model checkpoints
│   └── best_model.pth   # Pre-trained model
├── static/              # Web interface static files
├── outputs/             # Processed video outputs
└── requirements.txt     # Dependencies
```

## Configuration

### WhatsApp Alerts

To enable WhatsApp alerts:
1. Create a Twilio account
2. Set up a WhatsApp sandbox or get approval for production
3. Set the environment variables as shown in installation

### Model Configuration

The system uses a 3D CNN (C3D) architecture optimized for video analysis. You can:
- Adjust batch size in training for different GPU memory
- Modify learning rates and training epochs
- Add data augmentation techniques

## Performance

- **Accuracy**: Achieves >85% accuracy on the UCF Crime dataset
- **Processing Speed**: ~2-5 seconds per video (depending on length and hardware)
- **Memory Usage**: Optimized for 4GB+ GPU memory

## Requirements

- Python 3.7+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM
- 2GB+ GPU memory (for training)

## License

This project is for educational and research purposes. Please ensure compliance with local laws and regulations when using for surveillance applications.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues and questions, please create an issue in the GitHub repository.
