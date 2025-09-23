#!/usr/bin/env python3
"""
Demo Video Creator
Creates small synthetic videos for testing the surveillance system
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import argparse

def create_synthetic_video(output_path, duration=5, fps=30, width=224, height=224, video_type='normal'):
    """Create a synthetic video for testing"""
    
    # Calculate total frames
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Creating {video_type} video: {output_path}")
    
    for frame_idx in tqdm(range(total_frames), desc=f"Generating {video_type} video"):
        # Create base frame with noise
        frame = np.random.randint(50, 100, (height, width, 3), dtype=np.uint8)
        
        # Add patterns based on video type
        time_factor = frame_idx / total_frames
        
        if video_type == 'normal':
            # Normal activity - slow moving objects, calm colors
            # Add a moving circle (person walking)
            center_x = int(width * 0.2 + (width * 0.6) * time_factor)
            center_y = int(height * 0.7)
            cv2.circle(frame, (center_x, center_y), 20, (100, 150, 100), -1)
            
            # Add background elements
            cv2.rectangle(frame, (0, height-50), (width, height), (80, 120, 80), -1)
            
        elif video_type == 'fighting':
            # Fighting - rapid movements, red colors
            # Add two moving objects that collide
            obj1_x = int(width * 0.3 + (width * 0.2) * np.sin(frame_idx * 0.3))
            obj1_y = int(height * 0.5 + (height * 0.1) * np.cos(frame_idx * 0.2))
            
            obj2_x = int(width * 0.7 - (width * 0.2) * np.sin(frame_idx * 0.3))
            obj2_y = int(height * 0.5 - (height * 0.1) * np.cos(frame_idx * 0.2))
            
            # Make objects red and aggressive
            cv2.circle(frame, (obj1_x, obj1_y), 25, (0, 0, 200), -1)
            cv2.circle(frame, (obj2_x, obj2_y), 25, (0, 0, 200), -1)
            
            # Add motion blur effect
            if frame_idx > 0:
                frame = cv2.addWeighted(frame, 0.7, frame, 0.3, 0)
            
        elif video_type == 'robbery':
            # Robbery - quick movements, dark colors
            # Add a person running
            runner_x = int(width * time_factor)
            runner_y = int(height * 0.6)
            cv2.ellipse(frame, (runner_x, runner_y), (15, 30), 0, 0, 360, (50, 50, 50), -1)
            
            # Add something being taken (object disappears)
            if time_factor < 0.7:
                cv2.rectangle(frame, (int(width*0.8), int(height*0.3)), 
                            (int(width*0.9), int(height*0.4)), (0, 100, 200), -1)
            
        elif video_type == 'shooting':
            # Shooting - flashes, people running
            # Add muzzle flashes (random bright spots)
            if frame_idx % 10 < 3:  # Flash effect
                flash_x = int(width * 0.3)
                flash_y = int(height * 0.4)
                cv2.circle(frame, (flash_x, flash_y), 30, (255, 255, 255), -1)
            
            # Add people running away
            for i in range(3):
                person_x = int(width * (0.5 + 0.3 * time_factor + 0.1 * i))
                person_y = int(height * (0.7 + 0.1 * np.sin(frame_idx * 0.5 + i)))
                cv2.circle(frame, (person_x, person_y), 12, (100, 100, 150), -1)
        
        elif video_type == 'explosion':
            # Explosion - expanding bright circle
            explosion_radius = max(5, int(50 * time_factor + 10 * np.sin(frame_idx * 0.8)))
            center = (width // 2, height // 2)
            
            # Create explosion effect
            cv2.circle(frame, center, explosion_radius, (0, 100, 255), -1)
            cv2.circle(frame, center, explosion_radius + 10, (0, 200, 255), 5)
            
            # Add debris (random particles)
            debris_range = max(5, explosion_radius)
            for _ in range(20):
                debris_x = center[0] + np.random.randint(-debris_range, debris_range)
                debris_y = center[1] + np.random.randint(-debris_range, debris_range)
                cv2.circle(frame, (debris_x, debris_y), 2, (150, 150, 150), -1)
        
        elif video_type == 'vandalism':
            # Vandalism - spray patterns, defacing
            # Add spray paint effect
            spray_x = int(width * 0.3 + (width * 0.4) * time_factor)
            spray_y = int(height * 0.4)
            
            # Create spray pattern
            for _ in range(15):
                offset_x = np.random.randint(-20, 20)
                offset_y = np.random.randint(-20, 20)
                cv2.circle(frame, (spray_x + offset_x, spray_y + offset_y), 
                          np.random.randint(1, 4), (0, 255, 0), -1)
            
            # Add wall background
            cv2.rectangle(frame, (0, 0), (width, height//3), (150, 150, 150), -1)
        
        # Add timestamp
        timestamp = f"{frame_idx:04d}/{total_frames:04d}"
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
    
    # Release video writer
    out.release()
    print(f"Created: {output_path}")

def create_demo_video_set(output_dir='demo_videos'):
    """Create a set of demo videos for testing"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Video types to create
    video_types = [
        ('normal', 'Normal everyday activity'),
        ('fighting', 'Fighting/assault scenario'),
        ('robbery', 'Robbery in progress'), 
        ('shooting', 'Shooting incident'),
        ('explosion', 'Explosion event'),
        ('vandalism', 'Vandalism/graffiti')
    ]
    
    print("Creating demo video set for testing...")
    print("=" * 50)
    
    for video_type, description in video_types:
        output_path = os.path.join(output_dir, f'demo_{video_type}.mp4')
        print(f"\n{description}")
        create_synthetic_video(output_path, duration=8, fps=15, video_type=video_type)
    
    print(f"\nDemo videos created in: {output_dir}")
    print("You can use these videos to test the surveillance system:")
    print(f"  python app.py")
    print("  Then upload videos from the demo_videos folder")

def create_readme_for_demos(output_dir='demo_videos'):
    """Create a README for the demo videos"""
    readme_path = os.path.join(output_dir, 'README.md')
    
    content = """# Demo Videos

This directory contains synthetic demo videos for testing the Intelligent Video Surveillance System.

## Available Demo Videos

1. **demo_normal.mp4** - Normal everyday activity (person walking)
2. **demo_fighting.mp4** - Fighting/assault scenario with rapid movements
3. **demo_robbery.mp4** - Robbery in progress with quick movements
4. **demo_shooting.mp4** - Shooting incident with muzzle flashes
5. **demo_explosion.mp4** - Explosion event with expanding effects
6. **demo_vandalism.mp4** - Vandalism/graffiti with spray patterns

## How to Use

1. Start the web application:
   ```bash
   python app.py
   ```

2. Open your browser and go to `http://localhost:5000`

3. Upload any of the demo videos to see how the system detects different types of activities

4. The system will process the video and show:
   - Predicted activity type
   - Confidence score
   - Visualization with predictions overlaid
   - WhatsApp alerts (if configured)

## Note

These are synthetic videos created for demonstration purposes. The actual performance will be better with real surveillance footage after training on the UCF Crime dataset.

To get the full dataset and train your own model:
```bash
python download_dataset.py
python train.py --train_dir data/train --test_dir data/test
```
"""
    
    with open(readme_path, 'w') as f:
        f.write(content)
    
    print(f"Created README: {readme_path}")

def main():
    parser = argparse.ArgumentParser(description='Create demo videos for testing')
    parser.add_argument('--output_dir', type=str, default='demo_videos',
                       help='Output directory for demo videos')
    parser.add_argument('--duration', type=int, default=8,
                       help='Duration of each video in seconds')
    parser.add_argument('--fps', type=int, default=15,
                       help='Frames per second')
    parser.add_argument('--width', type=int, default=224,
                       help='Video width')
    parser.add_argument('--height', type=int, default=224,
                       help='Video height')
    
    args = parser.parse_args()
    
    # Create demo videos
    create_demo_video_set(args.output_dir)
    
    # Create README
    create_readme_for_demos(args.output_dir)
    
    print(f"\nDemo video set creation completed!")
    print(f"Videos saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
