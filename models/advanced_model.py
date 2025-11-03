import torch
import torch.nn as nn
import torch.nn.functional as F
class SpatialTemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SpatialTemporalAttention, self).__init__()
        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, None, None)),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.temporal_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply different types of attention
        spatial_att = self.spatial_attention(x)
        temporal_att = self.temporal_attention(x)
        channel_att = self.channel_attention(x)
        
        # Combine attentions
        x = x * spatial_att * temporal_att * channel_att
        return x

# Enhanced Residual Block with Attention (from train.py)
class EnhancedResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True):
        super(EnhancedResBlock3D, self).__init__()
        self.use_attention = use_attention
        
        # Main branch
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        
        # Attention mechanism
        if self.use_attention:
            self.attention = SpatialTemporalAttention(out_channels)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(0.1)

    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.use_attention:
            out = self.attention(out)
        
        out += identity
        out = self.relu(out)
        return out

# State-of-the-Art Enhanced C3D with Multiple Improvements (exact copy from train.py)
class MaxAccuracyC3D(nn.Module):
    def __init__(self, num_classes=14, dropout_rate=0.3, use_attention=True):
        super(MaxAccuracyC3D, self).__init__()
        self.use_attention = use_attention
        
        # Initial convolution with larger filters for better feature extraction
        self.stem = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        
        # Progressive feature extraction with residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 3, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
        
        # Global attention pooling
        self.global_attention = nn.Sequential(
            nn.Conv3d(512, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 1, 1),
            nn.Sigmoid()
        )
        
        # Adaptive pooling for variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Multi-scale feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        
        # Final classifier with label smoothing support
        self.classifier = nn.Linear(256, num_classes)
        
        self._initialize_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(EnhancedResBlock3D(in_channels, out_channels, stride, self.use_attention))
        
        for _ in range(1, blocks):
            layers.append(EnhancedResBlock3D(out_channels, out_channels, 1, self.use_attention))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Xavier/Kaiming initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Handle different input formats
        if len(x.shape) == 5:
            if x.size(2) == 3:  # (B, T, C, H, W) -> (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)
        
        # Stem processing
        x = self.stem(x)
        
        # Progressive feature extraction
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global attention weighted pooling
        attention_weights = self.global_attention(x)
        x = x * attention_weights
        
        # Global pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Feature fusion and classification
        x = self.feature_fusion(x)
        x = self.classifier(x)
        
        return x

# Keep the old name for compatibility
AdvancedVideoModel = MaxAccuracyC3D


# For compatibility with existing code, also provide C3D
class C3D(nn.Module):
    """
    Simple C3D model - fallback if advanced model doesn't work
    """
    def __init__(self, num_classes=14, pretrained=False):
        super(C3D, self).__init__()
        
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
        
        # Modified to match checkpoint architecture
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)  # Changed from 1024 to 4096
        self.fc8 = nn.Linear(4096, num_classes)  # Changed input from 1024 to 4096
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.init_weights()
        
    def init_weights(self):
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
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool3(x)
        
        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))
        x = self.pool4(x)
        
        x = F.relu(self.conv5a(x))
        x = F.relu(self.conv5b(x))
        x = self.pool5(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc6(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc7(x))
        x = self.dropout(x)
        
        x = self.fc8(x)
        
        return x
