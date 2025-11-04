import cv2
import numpy as np
import os

def extract_frames_from_video(video_path, sample_rate=10):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        sample_rate (int): Extract every Nth frame
        
    Returns:
        list: List of extracted frames as NumPy arrays
    """
    frames = []
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return frames
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    # Some files report 0 FPS; default to 25 for browser compatibility
    if not fps or fps <= 1e-3:
        fps = 25.0
    
    print(f"Video info: {frame_count} frames, {fps} FPS")
    
    # Extract frames
    for i in range(frame_count):
        # Set the frame position
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the frame
        ret, frame = video.read()
        
        if ret:
            # Only keep every Nth frame
            if i % sample_rate == 0:
                frames.append(frame)
        else:
            break
    
    # Release the video
    video.release()
    
    print(f"Extracted {len(frames)} frames")
    return frames

def process_video_frames(frames, clip_len=16, overlap=0.5):
    """
    Process extracted frames into clips
    
    Args:
        frames (list): List of video frames
        clip_len (int): Number of frames in each clip
        overlap (float): Overlap between consecutive clips (0-1)
        
    Returns:
        list: List of clips (each clip is a list of frames)
    """
    if len(frames) < clip_len:
        # If we have fewer frames than clip_len, duplicate frames
        frames = frames * (clip_len // len(frames) + 1)
        frames = frames[:clip_len]
        return [frames]
    
    # Calculate step size with overlap
    step_size = max(1, int(clip_len * (1 - overlap)))
    
    # Create clips
    clips = []
    for i in range(0, len(frames) - clip_len + 1, step_size):
        clip = frames[i:i + clip_len]
        clips.append(clip)
    
    print(f"Created {len(clips)} clips from {len(frames)} frames")
    return clips

def save_prediction_visualization(video_path, predictions, confidences, class_names, output_path=None):
    """
    Create a visualization of the predictions overlaid on the video
    
    Args:
        video_path (str): Path to the original video
        predictions (list): List of class predictions for each clip
        confidences (list): List of confidence values for each prediction
        class_names (list): List of class names
        output_path (str): Path to save the output video
        
    Returns:
        str: Path to the output video
    """
    # Set default output path if not provided
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = f"{base_name}_prediction.mp4"
    
    # Open the video
    video = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object with better browser compatibility
    # Try H.264 tag first ('avc1'), then fallback to 'mp4v'
    def _open_writer(path, fps, size):
        try:
            fourcc_try = cv2.VideoWriter_fourcc(*'avc1')
            vw = cv2.VideoWriter(path, fourcc_try, fps, size)
            if vw is not None and vw.isOpened():
                return vw
        except Exception:
            pass
        fourcc_fallback = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(path, fourcc_fallback, fps, size)

    out = _open_writer(output_path, fps, (width, height))
    if out is None or not out.isOpened():
        # As a last resort, try writing to a .avi container with MJPG
        alt_path = os.path.splitext(output_path)[0] + '.avi'
        fourcc_alt = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(alt_path, fourcc_alt, fps, (width, height))
        output_path = alt_path
    
    # Calculate frames per clip
    frames_per_clip = frame_count / len(predictions)
    
    # Process each frame
    frame_idx = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        
        # Calculate which clip this frame belongs to
        clip_idx = min(int(frame_idx / frames_per_clip), len(predictions) - 1)
        pred_class = predictions[clip_idx]
        confidence = confidences[clip_idx]
        
        # Add prediction text to frame
        text = f"{class_names[pred_class]}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Write the frame
        out.write(frame)
        
        frame_idx += 1
    
    # Release everything
    video.release()
    out.release()
    
    print(f"Visualization saved to {output_path}")
    return output_path


def export_anomaly_clip(video_path: str, center_frame: int, seconds: float = 3.0, out_dir: str = None,
                        prefix: str = "clip_", fallback_fps: float = 25.0) -> str:
    """Export a short MP4 clip around a center frame. Returns output filename (basename) if saved, else ''.
    The clip is written to out_dir (or same folder as video) and uses H.264 if available, else mp4v.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return ''
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 1e-3:
            fps = fallback_fps
        half = int(max(1, seconds * fps / 2))
        start = max(0, center_frame - half)
        end = min(total_frames - 1, center_frame + half)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        base_dir = out_dir if out_dir else os.path.dirname(video_path)
        os.makedirs(base_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        out_name = f"{prefix}{base_name}.mp4"
        out_path = os.path.join(base_dir, out_name)

        def _open_writer(path):
            try:
                f1 = cv2.VideoWriter_fourcc(*'avc1')
                vw = cv2.VideoWriter(path, f1, fps, (width, height))
                if vw is not None and vw.isOpened():
                    return vw
            except Exception:
                pass
            f2 = cv2.VideoWriter_fourcc(*'mp4v')
            return cv2.VideoWriter(path, f2, fps, (width, height))

        writer = _open_writer(out_path)
        if writer is None or not writer.isOpened():
            return ''

        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        cur = start
        while cur <= end:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            cur += 1

        cap.release()
        writer.release()
        return out_name
    except Exception:
        return ''