import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
# TensorBoard (optional)
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except Exception as _tb_err:
    _TB_AVAILABLE = False
    class SummaryWriter:  # fallback no-op writer
        ERR_MSG = str(_tb_err)
        def __init__(self, *args, **kwargs):
            print(f"WARNING: TensorBoard disabled ({self.ERR_MSG}). Continuing without TB logging.")
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
import argparse
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import cv2
import glob
from collections import OrderedDict, Counter
# Use new AMP API
import torch.amp as amp
import time
import json

# Label Smoothing Loss
class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label smoothing loss that prevents overconfidence.
    Instead of target being [0, 0, 1, 0, ...], it becomes [ε, ε, 1-ε, ε, ...]
    This helps with overfitting and better generalization.
    """
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        """
        Args:
            pred: Model predictions (logits) [batch_size, num_classes]
            target: Ground truth labels [batch_size]
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create smoothed labels
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
            # Apply class weights if provided
            if self.weight is not None:
                true_dist = true_dist * self.weight.unsqueeze(0)
        
        # Calculate loss
        loss = torch.sum(-true_dist * pred, dim=-1)
        return loss.mean()

# Define the C3D model
class C3D(nn.Module):
    def __init__(self, num_classes=14, dropout_rate=0.5):
        super(C3D, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # Fully connected layers
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        
        # Dropout layers
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # Batch normalization (optional, can improve training stability)
        self.use_bn = False  # Set to True to enable batch norm
        if self.use_bn:
            self.bn1 = nn.BatchNorm3d(64)
            self.bn2 = nn.BatchNorm3d(128)
            self.bn3a = nn.BatchNorm3d(256)
            self.bn3b = nn.BatchNorm3d(256)
            self.bn4a = nn.BatchNorm3d(512)
            self.bn4b = nn.BatchNorm3d(512)
            self.bn5a = nn.BatchNorm3d(512)
            self.bn5b = nn.BatchNorm3d(512)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Adjust input shape if needed (B, C, T, H, W)
        if x.size(2) == 3:
            x = x.permute(0, 2, 1, 3, 4)
        
        # Convolutional layers
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3a(x)
        if self.use_bn:
            x = self.bn3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        if self.use_bn:
            x = self.bn3b(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = self.conv4a(x)
        if self.use_bn:
            x = self.bn4a(x)
        x = F.relu(x)
        x = self.conv4b(x)
        if self.use_bn:
            x = self.bn4b(x)
        x = F.relu(x)
        x = self.pool4(x)
        
        x = self.conv5a(x)
        if self.use_bn:
            x = self.bn5a(x)
        x = F.relu(x)
        x = self.conv5b(x)
        if self.use_bn:
            x = self.bn5b(x)
        x = F.relu(x)
        x = self.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = self.fc8(x)
        
        return x

# Enhanced Dataset class with caching and optimization
class UCFCrimeDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None, max_clips_per_class=None, cache_frames=False):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transform = transform
        self.max_clips_per_class = max_clips_per_class
        self.cache_frames = cache_frames
        self.frame_cache = {} if cache_frames else None
        
        self.classes = OrderedDict()
        class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for i, class_name in enumerate(class_dirs):
            self.classes[class_name] = i
        
        self.clips = []
        self._find_clips()
    
    def _find_clips(self):
        total_images = 0
        print(f"Finding clips in {self.root_dir}...")
        
        for class_name, class_idx in self.classes.items():
            class_dir = os.path.join(self.root_dir, class_name)
            images = sorted(glob.glob(os.path.join(class_dir, '**', '*.png'), recursive=True))
            total_images += len(images)
            
            if len(images) == 0:
                print(f"WARNING: No images found for class {class_name}")
                continue
            
            # Group images into clips with overlap
            class_clips = []
            for i in range(0, len(images) - self.clip_len + 1, self.clip_len // 2):
                clip_paths = images[i:i + self.clip_len]
                if len(clip_paths) == self.clip_len:
                    class_clips.append((clip_paths, class_idx))
            
            # Limit clips per class if specified
            if self.max_clips_per_class and len(class_clips) > self.max_clips_per_class:
                class_clips = random.sample(class_clips, self.max_clips_per_class)
            
            self.clips.extend(class_clips)
            print(f"  {class_name}: {len(class_clips)} clips")
        
        print(f"Found {total_images} images in {self.root_dir}")
        print(f"Created {len(self.clips)} clips total")
    
    def __len__(self):
        return len(self.clips)
    
    def _load_frame(self, frame_path):
        """Load a single frame with optional caching"""
        if self.cache_frames and frame_path in self.frame_cache:
            return self.frame_cache[frame_path].copy()
        
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {frame_path}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.cache_frames:
            self.frame_cache[frame_path] = frame.copy()
        
        return frame
    
    def __getitem__(self, idx):
        clip_paths, label = self.clips[idx]
        clip = []
        
        for frame_path in clip_paths:
            try:
                frame = self._load_frame(frame_path)
                
                if self.transform:
                    frame = self.transform(frame)
                
                clip.append(frame)
            except Exception as e:
                print(f"Error loading frame {frame_path}: {e}")
                # Use a black frame as fallback
                if self.transform:
                    frame = self.transform(np.zeros((112, 112, 3), dtype=np.uint8))
                else:
                    frame = torch.zeros(3, 112, 112)
                clip.append(frame)
        
        # Stack frames to create a clip [C, T, H, W]
        clip = torch.stack(clip, dim=0).transpose(0, 1)
        
        return clip, label
    
    def get_class_distribution(self):
        """Get the distribution of classes in the dataset"""
        labels = [label for _, label in self.clips]
        return Counter(labels)

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop

def set_seed(seed):
    """Set seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """Create learning rate scheduler with warmup"""
    if args.warmup_epochs > 0:
        # Cosine annealing with warmup
        warmup_steps = args.warmup_epochs * steps_per_epoch
        total_steps = args.epochs * steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        # Standard ReduceLROnPlateau
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5
        )

def train_epoch(model, dataloader, criterion, optimizer, device, scaler, 
                accumulation_steps=1, clip_grad_norm=None, scheduler=None, epoch=0):
    """Train for one epoch with gradient accumulation and clipping"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}")
    for batch_idx, (clips, labels) in enumerate(progress_bar):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with amp.autocast('cuda'):
            outputs = model(clips)
            loss = criterion(outputs, labels)
            # Normalize loss for gradient accumulation
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Step scheduler if it's a step-based scheduler
            if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.LambdaLR):
                scheduler.step()
        
        # Statistics
        loss_value = loss.item() * accumulation_steps
        
        # Check for NaN loss
        if torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value)):
            print(f"\nWarning: NaN or Inf loss detected at batch {batch_idx}. Skipping batch.")
            optimizer.zero_grad()
            continue
        
        running_loss += loss_value
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    return running_loss / len(dataloader), 100. * correct / total

def validate(model, dataloader, criterion, device, train_dataset, max_batches=None):
    """Validate the model with optional batch limit for faster validation"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")
        for batch_idx, (clips, labels) in enumerate(progress_bar):
            if max_batches and batch_idx >= max_batches:
                break
                
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with amp.autocast('cuda'):
                outputs = model(clips)
                loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Save predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    # Calculate metrics
    class_names = list(train_dataset.classes.keys())
    # Use labels parameter to ensure all classes are included even if not predicted
    labels = list(range(len(class_names)))
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=labels)
    class_report = classification_report(all_labels, all_preds, labels=labels, target_names=class_names, zero_division=0)
    
    # Check prediction diversity
    unique_preds = len(set(all_preds))
    if unique_preds < len(class_names) // 2:
        print(f"\n⚠️  WARNING: Model only predicting {unique_preds}/{len(class_names)} classes!")
        print(f"   This suggests the model is collapsing to predicting majority classes.")
        from collections import Counter
        pred_dist = Counter(all_preds)
        print(f"   Top 3 predicted classes: {pred_dist.most_common(3)}")
    
    num_batches = max_batches if max_batches else len(dataloader)
    return running_loss / num_batches, 100. * correct / total, conf_matrix, class_report

def verify_dataset_structure(train_dir, test_dir):
    """Verify dataset structure and print helpful information"""
    print(f"\n=== Checking dataset structure ===")
    
    if not os.path.exists(train_dir):
        print(f"ERROR: Train directory {train_dir} does not exist!")
        return False
    
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    if not train_classes:
        print(f"ERROR: No class directories found in {train_dir}!")
        return False
    
    print(f"Found Train directory with {len(train_classes)} classes: {', '.join(train_classes)}")
    
    total_train_images = 0
    for class_name in train_classes:
        class_dir = os.path.join(train_dir, class_name)
        images = glob.glob(os.path.join(class_dir, '**', '*.png'), recursive=True)
        if not images:
            print(f"WARNING: No PNG images found in {class_dir}!")
        else:
            print(f"  - {class_name}: {len(images)} images")
            total_train_images += len(images)
    
    print(f"Total train images: {total_train_images}")
    
    if os.path.exists(test_dir):
        test_classes = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        if test_classes:
            print(f"Found Test directory with {len(test_classes)} classes.")
            total_test_images = 0
            for class_name in test_classes:
                class_dir = os.path.join(test_dir, class_name)
                images = glob.glob(os.path.join(class_dir, '**', '*.png'), recursive=True)
                if images:
                    print(f"  - {class_name}: {len(images)} images")
                    total_test_images += len(images)
            print(f"Total test images: {total_test_images}")
    else:
        print(f"WARNING: Test directory {test_dir} does not exist!")
    
    print("=== Dataset structure check complete ===\n")
    return True

def main(args):
    set_seed(args.seed)
    
    # Define paths (use provided args to support current project structure)
    train_dir = args.train_dir
    test_dir = args.test_dir
    
    # Verify dataset structure
    if not verify_dataset_structure(train_dir, test_dir):
        print("Dataset structure verification failed. Please fix the issues and try again.")
        return
    
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1024**3:.2f} GB")
    
    # Create transformations
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print(f"\nLoading training data from: {train_dir}")
    train_dataset = UCFCrimeDataset(
        root_dir=train_dir,
        clip_len=args.clip_len,
        transform=train_transform,
        max_clips_per_class=args.max_clips_per_class,
        cache_frames=args.cache_frames
    )
    
    print(f"\nLoading test data from: {test_dir}")
    val_dataset = UCFCrimeDataset(
        root_dir=test_dir,
        clip_len=args.clip_len,
        transform=val_transform,
        max_clips_per_class=args.max_val_clips_per_class,
        cache_frames=False
    )
    
    print(f"\nTrain dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("ERROR: Dataset is empty! Cannot continue training.")
        return
    
    # Get class distribution and compute weights
    train_dist = train_dataset.get_class_distribution()
    print(f"\nClass distribution in training set:")
    for class_name, class_idx in train_dataset.classes.items():
        count = train_dist.get(class_idx, 0)
        print(f"  {class_name}: {count} clips")
    
    # Compute class weights for handling imbalance
    if args.use_class_weights:
        labels = [label for _, label in train_dataset.clips]
        class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
        class_weights = torch.FloatTensor(class_weights).to(device)
        print(f"\nUsing class weights: {class_weights.cpu().numpy()}")
    else:
        class_weights = None
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create model
    model = C3D(num_classes=len(train_dataset.classes), dropout_rate=args.dropout_rate)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: C3D")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function with label smoothing and optional class weights
    if args.label_smoothing > 0:
        print(f"\nUsing Label Smoothing: {args.label_smoothing}")
        print("This prevents overconfidence and helps with overfitting.")
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(optimizer, args, len(train_loader))
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
    
    # TensorBoard writer
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(args.output_dir, 'training_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize GradScaler for mixed precision
    scaler = amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # Training loop
    best_acc = 0.0
    start_time = time.time()
    
    print(f"\n{'='*50}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*50}\n")
    
    try:
        for epoch in range(args.epochs):
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"{'-'*50}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler,
                accumulation_steps=args.accumulation_steps,
                clip_grad_norm=args.clip_grad_norm,
                scheduler=scheduler if isinstance(scheduler, optim.lr_scheduler.LambdaLR) else None,
                epoch=epoch
            )
            
            # Validate (with optional batch limit for faster validation)
            max_val_batches = args.max_val_batches if args.max_val_batches > 0 else None
            val_loss, val_acc, conf_matrix, class_report = validate(
                model, val_loader, criterion, device, train_dataset, max_val_batches
            )
            
            # Update learning rate (for ReduceLROnPlateau)
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            
            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            # Log to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
            writer.add_scalar('Time/epoch', epoch_time, epoch)
            
            # Print metrics
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            
            if (epoch + 1) % args.print_report_every == 0:
                print(f"\nDetailed Classification Report:")
                print(class_report)
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'class_names': list(train_dataset.classes.keys()),
            }, checkpoint_path)
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'best_acc': best_acc,
                    'class_names': list(train_dataset.classes.keys()),
                }, best_model_path)
                print(f"  ✓ New best model saved with accuracy: {best_acc:.2f}%")
            
            # Early stopping check
            if early_stopping(val_acc):
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation accuracy: {best_acc:.2f}%")
                break
            
            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"Error during training: {e}")
        print(f"{'='*50}")
        import traceback
        traceback.print_exc()
        print(f"\nSaving checkpoint before exiting...")
        
        # Save emergency checkpoint
        emergency_checkpoint = os.path.join(args.output_dir, 'emergency_checkpoint.pth')
        torch.save({
            'epoch': epoch + 1 if 'epoch' in locals() else 0,
            'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_names': list(train_dataset.classes.keys()),
        }, emergency_checkpoint)
        print(f"Emergency checkpoint saved to: {emergency_checkpoint}")
        writer.close()
        raise
    
    # Training complete
    total_time = time.time() - start_time
    writer.close()
    
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"{'='*50}")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Total training time: {total_time/3600:.2f} hours")
    print(f"Model saved to: {args.output_dir}")
    print(f"TensorBoard logs: {log_dir}")
    print(f"\nTo view training progress, run:")
    print(f"  tensorboard --logdir {log_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced C3D training for UCF Crime dataset")
    
    # Data parameters
    parser.add_argument('--train_dir', type=str, default='data/Train', 
                        help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, default='data/Test', 
                        help='Path to test data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', 
                        help='Directory to save models and logs')
    parser.add_argument('--clip_len', type=int, default=16, 
                        help='Number of frames in each clip')
    parser.add_argument('--max_clips_per_class', type=int, default=None, 
                        help='Maximum clips per class (for faster training/debugging)')
    parser.add_argument('--max_val_clips_per_class', type=int, default=None, 
                        help='Maximum validation clips per class')
    parser.add_argument('--cache_frames', action='store_true', 
                        help='Cache frames in memory (requires lots of RAM)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size per GPU')
    parser.add_argument('--accumulation_steps', type=int, default=4, 
                        help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, 
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, 
                        help='Dropout rate')
    
    # Optimization parameters
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, 
                        help='Gradient clipping max norm (None to disable)')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='Number of warmup epochs (0 to disable)')
    parser.add_argument('--use_class_weights', action='store_true', 
                        help='Use class weights to handle imbalance')
    parser.add_argument('--label_smoothing', type=float, default=0.0, 
                        help='Label smoothing factor (0.0 to 1.0, recommended: 0.1). Prevents overconfidence and overfitting.')
    
    # Early stopping
    parser.add_argument('--early_stopping_patience', type=int, default=10, 
                        help='Early stopping patience (epochs)')
    
    # Validation parameters
    parser.add_argument('--max_val_batches', type=int, default=0, 
                        help='Max validation batches per epoch (0 for full validation)')
    parser.add_argument('--print_report_every', type=int, default=5, 
                        help='Print detailed classification report every N epochs')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of dataloader workers')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID to use')
    parser.add_argument('--multi_gpu', action='store_true', 
                        help='Use multiple GPUs with DataParallel')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*50)
    print("Training Configuration")
    print("="*50)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:.<30} {value}")
    print("="*50 + "\n")
    
    # Run training
    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
