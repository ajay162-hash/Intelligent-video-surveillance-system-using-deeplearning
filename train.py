#!/usr/bin/env python3
"""
Main training script for Intelligent Video Surveillance System
Memory-optimized for efficient training and maximum accuracy
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import glob
from collections import OrderedDict
import random
import json
import time
from datetime import datetime
import gc

# Import model architecture
from models.cnn3d import C3D

# Set random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_random_seeds(42)

class VideoDataset(Dataset):
    """Dataset for video crime detection training"""
    def __init__(self, root_dir, clip_len=16, max_clips_per_class=400, is_training=True):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.max_clips_per_class = max_clips_per_class
        self.is_training = is_training
        
        # Define class names (must match the order used in your trained model)
        class_names = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary', 
                      'Explosion', 'Fighting', 'Normal', 'RoadAccidents', 
                      'Robbery', 'Shooting', 'Shoplifting', 'Stealing', 'Vandalism']
        
        self.classes = OrderedDict()
        for i, class_name in enumerate(class_names):
            self.classes[class_name] = i
        
        # Data augmentation for training
        if is_training:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.RandomCrop((112, 112)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        self.clips = []
        self._create_clips()
    
    def _create_clips(self):
        """Create clips from processed frames"""
        print(f"Creating dataset from {self.root_dir}")
        
        if not os.path.exists(self.root_dir):
            print(f"ERROR: Dataset directory not found: {self.root_dir}")
            return
        
        for class_name, class_idx in tqdm(self.classes.items(), desc="Processing classes"):
            class_dir = os.path.join(self.root_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f"WARNING: Class directory not found: {class_dir}")
                continue
            
            # Get all PNG images
            images = glob.glob(os.path.join(class_dir, "*.png"))
            images = sorted(images)
            
            if len(images) < self.clip_len:
                print(f"WARNING: {class_name} has only {len(images)} images, need at least {self.clip_len}")
                continue
            
            # Smart sampling - vary stride based on video length
            if len(images) > 10000:  # Long videos
                stride = max(self.clip_len, len(images) // self.max_clips_per_class)
            elif len(images) > 2000:  # Medium videos
                stride = self.clip_len // 2
            else:  # Short videos
                stride = max(1, self.clip_len // 4)
            
            clips_created = 0
            
            for start_idx in range(0, len(images) - self.clip_len + 1, stride):
                if clips_created >= self.max_clips_per_class:
                    break
                    
                clip_paths = images[start_idx:start_idx + self.clip_len]
                if len(clip_paths) == self.clip_len:
                    self.clips.append((clip_paths, class_idx))
                    clips_created += 1
            
            print(f"  {class_name}: {clips_created} clips from {len(images)} images")
        
        print(f"Total clips: {len(self.clips)}")
    
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, idx):
        clip_paths, label = self.clips[idx]
        
        frames = []
        for path in clip_paths:
            image = cv2.imread(path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.transform(image)
                frames.append(image)
        
        if len(frames) != self.clip_len:
            # Pad with last frame if needed
            while len(frames) < self.clip_len:
                frames.append(frames[-1] if frames else torch.zeros(3, 112, 112))
        
        # Stack frames and reorder dimensions to [C, T, H, W]
        clip = torch.stack(frames)  # [T, C, H, W]
        clip = clip.permute(1, 0, 2, 3)  # [C, T, H, W]
        
        return clip, label

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Clear cache to prevent memory buildup
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    return running_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(val_loader, desc='Validation'):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    
    return val_loss, accuracy

def main():
    parser = argparse.ArgumentParser(description='Video Crime Detection Training')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to testing data directory')
    parser.add_argument('--output_dir', type=str, default='trained_models', help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (reduce if OOM)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--clip_len', type=int, default=16, help='Number of frames per clip')
    parser.add_argument('--max_clips', type=int, default=400, help='Maximum clips per class')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = VideoDataset(args.train_dir, args.clip_len, args.max_clips, is_training=True)
    val_dataset = VideoDataset(args.test_dir, args.clip_len, args.max_clips, is_training=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: No data found! Please check your data directories.")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    # Create model
    print("Creating model...")
    model = C3D(num_classes=14).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)
    
    # Training loop
    best_acc = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        # Clean up GPU memory
        torch.cuda.empty_cache()
    
    # Training completed
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/3600:.2f} hours!")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_acc': best_acc,
        'total_time_hours': total_time/3600
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Training history saved to {args.output_dir}/training_history.json")

if __name__ == '__main__':
    main()
