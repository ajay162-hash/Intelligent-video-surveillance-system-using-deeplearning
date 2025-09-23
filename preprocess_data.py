#!/usr/bin/env python3
"""
Manual Data Preprocessing Script for UCF Crime Dataset
Converts video files to frame sequences for training
"""

import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil

# Class names expected by the model
CLASS_NAMES = [
    'Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
    'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 'Robbery', 
    'Shooting', 'Shoplifting', 'Stealing', 'Vandalism'
]

def extract_frames_from_video(video_path, output_dir, sample_rate=10):
    """Extract frames from a single video file"""
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return False
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps if fps > 0 else 0
    
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"  Duration: {duration:.1f}s, Frames: {frame_count}, FPS: {fps:.1f}")
    
    frame_idx = 0
    saved_count = 0
    
    # Extract frames with progress bar
    with tqdm(total=frame_count, desc=f"Extracting frames") as pbar:
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
            pbar.update(1)
    
    cap.release()
    print(f"  Extracted {saved_count} frames")
    return True

def process_video_directory(input_dir, output_dir, sample_rate=10):
    """Process all videos in a directory structure"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    processed_count = 0
    
    print(f"Scanning directory: {input_dir}")
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Check if file is a video
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_path = os.path.join(root, file)
                
                # Determine class name from directory structure
                rel_path = os.path.relpath(root, input_dir)
                path_parts = rel_path.split(os.sep)
                
                # Try to find class name in path
                class_name = None
                for part in path_parts:
                    if part in CLASS_NAMES:
                        class_name = part
                        break
                
                # If no class found in path, try to infer from filename or parent directory
                if not class_name:
                    # Check if parent directory name matches a class
                    parent_dir = os.path.basename(root)
                    if parent_dir in CLASS_NAMES:
                        class_name = parent_dir
                    else:
                        # Default to filename-based classification
                        filename_lower = file.lower()
                        for cls in CLASS_NAMES:
                            if cls.lower() in filename_lower:
                                class_name = cls
                                break
                        
                        if not class_name:
                            class_name = 'Unknown'
                
                print(f"\nFound video: {file} -> Class: {class_name}")
                
                # Create output directory for this video
                video_name = os.path.splitext(file)[0]
                class_output_dir = os.path.join(output_dir, class_name)
                video_output_dir = os.path.join(class_output_dir, video_name)
                
                # Extract frames
                if extract_frames_from_video(video_path, video_output_dir, sample_rate):
                    processed_count += 1
    
    print(f"\nProcessed {processed_count} videos total")
    return processed_count

def create_train_test_split(data_dir, train_ratio=0.8):
    """Split processed data into train and test sets"""
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"\nCreating train/test split (train ratio: {train_ratio})")
    
    total_train = 0
    total_test = 0
    
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        # Get all video directories for this class
        video_dirs = [d for d in os.listdir(class_dir) 
                     if os.path.isdir(os.path.join(class_dir, d))]
        
        if not video_dirs:
            print(f"No videos found for class: {class_name}")
            continue
        
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
        train_frames = 0
        test_frames = 0
        
        for video_dir in train_videos:
            src = os.path.join(class_dir, video_dir)
            for frame_file in os.listdir(src):
                if frame_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src, frame_file),
                        os.path.join(train_class_dir, f"{video_dir}_{frame_file}")
                    )
                    train_frames += 1
        
        for video_dir in test_videos:
            src = os.path.join(class_dir, video_dir)
            for frame_file in os.listdir(src):
                if frame_file.endswith('.png'):
                    shutil.copy2(
                        os.path.join(src, frame_file),
                        os.path.join(test_class_dir, f"{video_dir}_{frame_file}")
                    )
                    test_frames += 1
        
        total_train += train_frames
        total_test += test_frames
        
        print(f"{class_name}: {len(train_videos)} train videos ({train_frames} frames), "
              f"{len(test_videos)} test videos ({test_frames} frames)")
    
    print(f"\nTotal: {total_train} train frames, {total_test} test frames")

def cleanup_intermediate_directories(data_dir):
    """Clean up intermediate class directories after train/test split"""
    answer = input("Remove intermediate class directories? (y/N): ").lower()
    if answer == 'y':
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                shutil.rmtree(class_dir)
                print(f"Removed: {class_dir}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess videos for training')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing video files')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory for processed frames')
    parser.add_argument('--sample_rate', type=int, default=10,
                       help='Extract every Nth frame from videos')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training (0.0-1.0)')
    parser.add_argument('--no_split', action='store_true',
                       help='Skip train/test split')
    parser.add_argument('--cleanup', action='store_true',
                       help='Cleanup intermediate directories after processing')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory does not exist: {args.input_dir}")
        return
    
    if not (0.0 <= args.train_ratio <= 1.0):
        print("Error: train_ratio must be between 0.0 and 1.0")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("UCF Crime Dataset Preprocessor")
    print("=" * 40)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample rate: every {args.sample_rate} frames")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Expected classes: {', '.join(CLASS_NAMES)}")
    print()
    
    # Process videos
    processed_count = process_video_directory(args.input_dir, args.output_dir, args.sample_rate)
    
    if processed_count == 0:
        print("No videos were processed. Please check:")
        print("1. Input directory contains video files (.mp4, .avi, .mov, etc.)")
        print("2. Video files are accessible")
        print("3. Directory structure matches expected format")
        return
    
    # Create train/test split
    if not args.no_split:
        create_train_test_split(args.output_dir, args.train_ratio)
        
        # Cleanup intermediate directories
        if args.cleanup:
            cleanup_intermediate_directories(args.output_dir)
    
    print("\nPreprocessing completed!")
    print(f"Data saved to: {args.output_dir}")
    
    # Display summary
    train_dir = os.path.join(args.output_dir, 'train')
    test_dir = os.path.join(args.output_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        print("\nDataset structure:")
        for split in ['train', 'test']:
            split_dir = os.path.join(args.output_dir, split)
            if os.path.exists(split_dir):
                classes = [d for d in os.listdir(split_dir) 
                          if os.path.isdir(os.path.join(split_dir, d))]
                print(f"  {split}: {len(classes)} classes")
                for class_name in classes:
                    class_dir = os.path.join(split_dir, class_name)
                    frame_count = len([f for f in os.listdir(class_dir) 
                                     if f.endswith('.png')])
                    print(f"    {class_name}: {frame_count} frames")
    
    print(f"\nReady for training! Run:")
    print(f"python train.py --train_dir {args.output_dir}/train --test_dir {args.output_dir}/test")

if __name__ == "__main__":
    main()
