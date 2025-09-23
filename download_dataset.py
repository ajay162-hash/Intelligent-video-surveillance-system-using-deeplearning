#!/usr/bin/env python3
"""
Automatic UCF Crime Dataset Downloader and Preprocessor
Downloads the UCF Crime dataset and preprocesses it for training
"""

import os
import requests
import zipfile
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import shutil
from urllib.parse import urlparse
import tarfile
import gzip

# UCF Crime Dataset URLs (using alternative download sources)
DATASET_URLS = [
    # Primary source (if available)
    "http://crcv.ucf.edu/projects/real-world/Anomaly_Detection_in_Video_files.tar.gz",
    # Alternative sources - these are placeholders, you'll need to update with actual URLs
    # "https://www.dropbox.com/s/example/ucf_crime_dataset.zip",
    # "https://drive.google.com/file/d/example/view?usp=sharing",
]

# Class names and their expected folders
CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

def download_file(url, destination):
    """Download a file with progress bar"""
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
    except Exception as e:
        print(f"Failed to download {url}: {str(e)}")
        return False

def extract_archive(archive_path, extract_to):
    """Extract various types of archives"""
    try:
        if archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(extract_to)
        elif archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.endswith('.gz'):
            with gzip.open(archive_path, 'rb') as f_in:
                with open(archive_path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        else:
            print(f"Unsupported archive format: {archive_path}")
            return False
        
        return True
    except Exception as e:
        print(f"Failed to extract {archive_path}: {str(e)}")
        return False

def extract_frames_from_video(video_path, output_dir, sample_rate=10):
    """Extract frames from a video file"""
    if not os.path.exists(video_path):
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    saved_count = 0
    
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save every nth frame
        if frame_idx % sample_rate == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {os.path.basename(video_path)}")
    return True

def process_videos_in_directory(input_dir, output_dir, sample_rate=10):
    """Process all videos in a directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                
                # Determine class name from directory structure
                rel_path = os.path.relpath(root, input_dir)
                class_name = rel_path.split(os.sep)[0] if rel_path != '.' else 'Unknown'
                
                # Create output directory for this class
                class_output_dir = os.path.join(output_dir, class_name)
                video_output_dir = os.path.join(class_output_dir, os.path.splitext(file)[0])
                
                print(f"Processing {video_path} -> {video_output_dir}")
                extract_frames_from_video(video_path, video_output_dir, sample_rate)

def create_train_test_split(data_dir, train_ratio=0.8):
    """Split data into train and test sets"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        # Get all video directories for this class
        video_dirs = [d for d in os.listdir(class_dir) 
                     if os.path.isdir(os.path.join(class_dir, d))]
        
        # Shuffle and split
        np.random.shuffle(video_dirs)
        split_idx = int(len(video_dirs) * train_ratio)
        
        train_videos = video_dirs[:split_idx]
        test_videos = video_dirs[split_idx:]
        
        # Create class directories in train/test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy frames to appropriate directories
        for video_dir in train_videos:
            src = os.path.join(class_dir, video_dir)
            for frame_file in os.listdir(src):
                if frame_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src, frame_file),
                        os.path.join(train_class_dir, f"{video_dir}_{frame_file}")
                    )
        
        for video_dir in test_videos:
            src = os.path.join(class_dir, video_dir)
            for frame_file in os.listdir(src):
                if frame_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src, frame_file),
                        os.path.join(test_class_dir, f"{video_dir}_{frame_file}")
                    )
        
        print(f"{class_name}: {len(train_videos)} train, {len(test_videos)} test videos")

def create_sample_dataset():
    """Create a small sample dataset for testing if main dataset is not available"""
    print("Creating sample dataset for testing...")
    
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create train and test directories
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        # Create a few sample classes
        sample_classes = ['Normal', 'Fighting', 'Robbery', 'Shooting']
        
        for class_name in sample_classes:
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Create some dummy images (colored rectangles)
            num_samples = 50 if split == 'train' else 20
            for i in range(num_samples):
                # Create a random colored image
                img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
                
                # Add some pattern based on class
                if class_name == 'Fighting':
                    img[:, :, 0] = 200  # More red
                elif class_name == 'Robbery':
                    img[:, :, 1] = 200  # More green
                elif class_name == 'Shooting':
                    img[:, :, 2] = 200  # More blue
                
                cv2.imwrite(os.path.join(class_dir, f"sample_{i:03d}.png"), img)
    
    print("Sample dataset created in 'data/' directory")
    return True

def main():
    parser = argparse.ArgumentParser(description='Download and preprocess UCF Crime dataset')
    parser.add_argument('--output_dir', type=str, default='data', 
                       help='Output directory for processed data')
    parser.add_argument('--sample_rate', type=int, default=10,
                       help='Extract every Nth frame from videos')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--sample_only', action='store_true',
                       help='Create sample dataset only (for testing)')
    
    args = parser.parse_args()
    
    if args.sample_only:
        create_sample_dataset()
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Try to download the dataset
    download_success = False
    archive_path = None
    
    print("Attempting to download UCF Crime dataset...")
    print("Note: The original UCF dataset may require manual download due to access restrictions.")
    print("If automatic download fails, please:")
    print("1. Visit: http://crcv.ucf.edu/projects/real-world/")
    print("2. Download the 'Anomaly Detection in Videos' dataset manually")
    print("3. Extract it and run: python preprocess_data.py --input_dir /path/to/videos")
    print()
    
    for url in DATASET_URLS:
        try:
            filename = os.path.basename(urlparse(url).path)
            if not filename:
                filename = "ucf_crime_dataset.tar.gz"
            
            archive_path = os.path.join(args.output_dir, filename)
            
            if download_file(url, archive_path):
                download_success = True
                break
        except Exception as e:
            print(f"Download failed: {e}")
            continue
    
    if download_success and archive_path:
        print("Extracting dataset...")
        if extract_archive(archive_path, args.output_dir):
            print("Dataset extracted successfully")
            
            # Find video directory
            video_dir = None
            for root, dirs, files in os.walk(args.output_dir):
                if any(f.endswith(('.mp4', '.avi', '.mov')) for f in files):
                    video_dir = root
                    break
            
            if video_dir:
                print("Processing videos...")
                process_videos_in_directory(video_dir, args.output_dir, args.sample_rate)
                
                print("Creating train/test split...")
                create_train_test_split(args.output_dir, args.train_ratio)
                
                print("Dataset setup completed!")
            else:
                print("No video files found in extracted dataset")
                print("Creating sample dataset for testing...")
                create_sample_dataset()
        
        # Clean up archive
        if os.path.exists(archive_path):
            os.remove(archive_path)
    else:
        print("Could not download dataset automatically.")
        print("Creating sample dataset for testing purposes...")
        create_sample_dataset()
        print("\nTo use the full dataset:")
        print("1. Download UCF Crime dataset manually")
        print("2. Run: python preprocess_data.py --input_dir /path/to/videos --output_dir data")

if __name__ == "__main__":
    main()
