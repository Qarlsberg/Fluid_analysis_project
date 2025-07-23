"""
RefineNet Model for Fluid and Slag Segmentation with Quality Classification

This module contains the neural network architecture for detecting and segmenting
fluid and slag in industrial imagery while also classifying image quality.

Key Concepts for Data Engineers:
- Segmentation: Pixel-level classification to identify which parts of an image
  contain specific materials (fluid, slag)
- Quality Classification: Determining if an image is clear, steamy, or fuzzy
- ROI (Region of Interest): A specific rectangular area within an image where
  analysis is focused
- Neural Network: A machine learning model inspired by biological neural networks
- Convolutional Neural Network (CNN): Specialized for processing image data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from typing import Dict

# Quality readability mapping - determines how reliable measurements are
# based on image quality conditions
QUALITY_READABILITY = {
    0: 1.0,    # Clear: fully readable and reliable measurements
    1: 0.4,    # Steam: partially readable, measurements less reliable
    2: 0.7     # Fuzzy: poorly readable, measurements should be used with caution
}

class ResidualConv(nn.Module):
    """
    Residual Convolution Block for Feature Processing
    
    A building block that helps the neural network learn better by allowing
    information to "skip" through layers. This prevents the vanishing gradient
    problem in deep networks.
    
    Key Concepts:
    - Residual Connection: Adds the input directly to the output, helping
      preserve information flow
    - Convolution: A mathematical operation that detects features like edges
      and patterns in images
    - Batch Normalization: Normalizes input data to stabilize training
    - Dropout: Randomly sets some neurons to zero during training to prevent
      overfitting
    
    Args:
        in_channels (int): Number of input feature channels
        out_channels (int): Number of output feature channels
        dropout (float): Probability of dropping neurons (0.0-1.0)
    """
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = None
        
    def forward(self, x):
        """
        Forward pass through the residual block
        
        Args:
            x (torch.Tensor): Input feature tensor
            
        Returns:
            torch.Tensor: Processed features with residual connection
        """
        # Store original input for residual connection
        residual = x
        
        # First convolution + normalization + activation
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)  # Apply dropout for regularization
        
        # Second convolution + normalization (no activation yet)
        out = self.bn2(self.conv2(out))
        
        # Apply skip connection if channels don't match
        if self.skip is not None:
            residual = self.skip(residual)
        
        # Add residual connection and apply final activation
        out += residual  # This is the "residual" part
        out = F.relu(out, inplace=True)
        return out

class QualityClassifier(nn.Module):
    """
    Image Quality Classifier Component
    
    This component analyzes image features to determine the quality of the input image.
    It classifies images into three categories: Clear, Steam, or Fuzzy.
    
    Key Concepts:
    - Global Average Pooling: Reduces spatial dimensions by averaging across the
      entire feature map, creating a fixed-size representation
    - Classification Head: Final layers that convert features into class predictions
    - Dropout: Regularization technique that randomly ignores some neurons to
      prevent overfitting
    
    Quality Classes:
    - 0 (Clear): High quality image with good visibility
    - 1 (Steam): Image obscured by steam, reduced visibility
    - 2 (Fuzzy): Blurry or low quality image
    
    Args:
        in_channels (int): Number of input feature channels from backbone
        num_classes (int): Number of quality classes (default: 3)
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, in_channels: int, num_classes: int = 3, dropout: float = 0.5):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        reduced_channels = in_channels // 4
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, reduced_channels, bias=False),
            nn.BatchNorm1d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(reduced_channels, num_classes)
        )
        
    def forward(self, x):
        """
        Classify image quality from feature maps
        
        Args:
            x (torch.Tensor): Input feature maps from backbone network
            
        Returns:
            torch.Tensor: Quality class logits (raw predictions before softmax)
        """
        # Global average pooling: Convert 2D feature maps to 1D vector
        x = self.gap(x)  # Shape: [batch, channels, 1, 1]
        
        # Flatten to 1D feature vector
        x = x.flatten(1)  # Shape: [batch, channels]
        
        # Apply classifier to get quality predictions
        return self.classifier(x)  # Shape: [batch, num_classes]

class RefinementBlock(nn.Module):
    """
    Feature Refinement Block for Multi-Scale Feature Fusion
    
    This block combines high-level semantic features (what objects are present)
    with low-level detail features (edges, textures) to create refined feature
    representations. It's a key component of the RefineNet architecture.
    
    Key Concepts:
    - Multi-scale Fusion: Combining features from different resolution levels
    - High-level Features: Abstract features that capture semantic information
      (e.g., "this is fluid")
    - Low-level Features: Detailed features that capture fine-grained information
      (e.g., edges, textures)
    - Attention Mechanisms: Help the model focus on important features
    - Channel Attention: Emphasizes important feature channels
    - Spatial Attention: Emphasizes important spatial locations
    
    Args:
        high_channels (int): Number of channels in high-level features
        low_channels (int): Number of channels in low-level features
        out_channels (int): Number of output channels
        dropout (float): Dropout rate for regularization
    """
    def __init__(self, high_channels: int, low_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        # Adjust input channels for residual block
        self.residual = ResidualConv(low_channels, out_channels, dropout)
        reduced_channels = max(out_channels // 8, 8)
        
        # Efficient channel attention
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, reduced_channels, 1, bias=False),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # Projection for high-level features if needed
        if high_channels != out_channels:
            self.high_proj = nn.Conv2d(high_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.high_proj = None
        
    def forward(self, high_feat, low_feat):
        """
        Fuse high-level and low-level features with attention mechanisms
        
        Args:
            high_feat (torch.Tensor): High-level semantic features (lower resolution)
            low_feat (torch.Tensor): Low-level detail features (higher resolution)
            
        Returns:
            torch.Tensor: Refined features combining both inputs
        """
        # Project high-level features to match output channels if needed
        if self.high_proj is not None:
            high_feat = self.high_proj(high_feat)
        
        # Upsample low-level features to match high-level feature spatial size
        # This aligns the feature maps for element-wise operations
        low_feat = F.interpolate(
            low_feat,
            size=high_feat.shape[2:],  # Match height and width
            mode='bilinear',           # Smooth upsampling
            align_corners=False
        )
        
        # Apply residual convolution to refine low-level features
        refined = self.residual(low_feat)
        
        # Apply channel attention: "Which feature channels are important?"
        channel_att = self.channel_gate(refined)  # Attention weights [0,1]
        refined = refined * channel_att           # Weighted features
        
        # Apply spatial attention: "Which spatial locations are important?"
        spatial_att = self.spatial_gate(refined)  # Attention weights [0,1]
        refined = refined * spatial_att           # Weighted features
        
        # Combine high-level semantic info with refined low-level details
        out = high_feat + refined  # Element-wise addition
        return out

class EnhancedRefineNet(nn.Module):
    """
    Enhanced RefineNet for Fluid and Slag Segmentation with Quality Classification
    
    This is the main neural network model that performs three tasks:
    1. Fluid Segmentation: Identifies pixels containing fluid
    2. Slag Segmentation: Identifies pixels containing slag
    3. Quality Classification: Determines image quality (Clear/Steam/Fuzzy)
    
    Architecture Overview:
    - Backbone: ResNet-34 encoder for feature extraction
    - Decoder: RefineNet-style decoder with multi-scale feature fusion
    - Segmentation Heads: Separate outputs for fluid and slag masks
    - Quality Head: Classifier for image quality assessment
    
    Key Concepts for Data Engineers:
    - Segmentation: Pixel-level classification (each pixel gets a label)
    - Encoder-Decoder: Common architecture pattern where:
      * Encoder: Extracts high-level features (downsampling)
      * Decoder: Reconstructs spatial details (upsampling)
    - Multi-task Learning: Training one model for multiple related tasks
    - Transfer Learning: Using pre-trained weights from ImageNet
    
    Args:
        config (Dict): Configuration dictionary containing model parameters
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.dropout = config['model']['dropout']
        
        # Get channel configurations
        enc_channels = config['model']['channels']['encoder']
        dec_channels = config['model']['channels']['decoder']
        
        # ResNet34 backbone
        if config['model']['pretrained']:
            backbone = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            backbone = resnet34(weights=None)
            
        # Encoder
        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels
        self.layer4 = backbone.layer4  # 512 channels
        
        # Quality classifier
        self.quality_classifier = QualityClassifier(enc_channels[-1], num_classes=3)
        
        # Refinement path with memory-efficient blocks
        self.refine4 = RefinementBlock(enc_channels[3], enc_channels[3], dec_channels[0], self.dropout)
        self.refine3 = RefinementBlock(enc_channels[2], dec_channels[0], dec_channels[1], self.dropout)
        self.refine2 = RefinementBlock(enc_channels[1], dec_channels[1], dec_channels[2], self.dropout)
        self.refine1 = RefinementBlock(enc_channels[0], dec_channels[2], dec_channels[3], self.dropout)
        
        # Deep supervision heads
        if config['model']['aux_loss']:
            self.aux_head4 = nn.Sequential(
                nn.Conv2d(dec_channels[0], 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.aux_head3 = nn.Sequential(
                nn.Conv2d(dec_channels[1], 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
            self.aux_head2 = nn.Sequential(
                nn.Conv2d(dec_channels[2], 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # Output heads with improved architecture
        self.fluid_head = nn.Sequential(
            ResidualConv(dec_channels[3], 32, self.dropout),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.slag_head = nn.Sequential(
            ResidualConv(dec_channels[3], 32, self.dropout),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the RefineNet for inference
        
        This method processes an input image through the entire network to produce:
        1. Fluid segmentation mask
        2. Slag segmentation mask
        3. Image quality classification
        
        Data Flow:
        1. Encoder: Extract multi-scale features using ResNet backbone
        2. Quality Classification: Classify image quality from deepest features
        3. Decoder: Refine and fuse features from coarse to fine resolution
        4. Segmentation Heads: Generate final masks for fluid and slag
        
        Args:
            x (torch.Tensor): Input image tensor with shape [batch, 3, H, W]
                             Values should be normalized to [0, 1] range
            
        Returns:
            dict: Dictionary containing:
                - 'fluid_mask': Tensor with fluid segmentation (shape: [batch, 1, H, W])
                - 'slag_mask': Tensor with slag segmentation (shape: [batch, 1, H, W])
                - 'quality': Tensor with quality logits (shape: [batch, 3])
                - 'aux*': Optional auxiliary outputs for deep supervision
        """
        # Store original input dimensions for final upsampling
        original_height, original_width = x.shape[2:]
        
        # === ENCODER PATH: Extract hierarchical features ===
        # Each layer progressively downsamples and extracts higher-level features
        x = self.layer0(x)     # Initial conv+pool: H/4 x W/4, 64 channels
        x1 = self.layer1(x)    # ResNet block 1: H/4 x W/4, 64 channels
        x2 = self.layer2(x1)   # ResNet block 2: H/8 x W/8, 128 channels
        x3 = self.layer3(x2)   # ResNet block 3: H/16 x W/16, 256 channels
        x4 = self.layer4(x3)   # ResNet block 4: H/32 x W/32, 512 channels
        
        # === QUALITY CLASSIFICATION ===
        # Use deepest features (x4) which contain the most semantic information
        quality_prediction = self.quality_classifier(x4)
        
        # === DECODER PATH: Progressive Feature Refinement ===
        auxiliary_outputs = {}
        
        if self.config['model']['aux_loss']:
            # Deep supervision mode: Generate intermediate predictions for training
            r4 = self.refine4(x4, x4)  # Refine deepest features
            auxiliary_outputs['aux4'] = F.interpolate(
                self.aux_head4(r4),
                size=(original_height, original_width),
                mode='bilinear', align_corners=False
            )
            
            r3 = self.refine3(x3, r4)  # Fuse layer3 features with refined r4
            auxiliary_outputs['aux3'] = F.interpolate(
                self.aux_head3(r3),
                size=(original_height, original_width),
                mode='bilinear', align_corners=False
            )
            
            r2 = self.refine2(x2, r3)  # Fuse layer2 features with refined r3
            auxiliary_outputs['aux2'] = F.interpolate(
                self.aux_head2(r2),
                size=(original_height, original_width),
                mode='bilinear', align_corners=False
            )
            
            r1 = self.refine1(x1, r2)  # Final refinement with layer1 features
        else:
            # Standard mode: Only main prediction path
            r4 = self.refine4(x4, x4)  # Start refinement from deepest level
            r3 = self.refine3(x3, r4)  # Progressively add finer details
            r2 = self.refine2(x2, r3)  # Continue adding spatial detail
            r1 = self.refine1(x1, r2)  # Final high-resolution features
        
        # Upsample final features to match input resolution
        refined_features = F.interpolate(
            r1,
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )
        
        # === SEGMENTATION HEADS: Generate Final Masks ===
        # Apply separate heads for fluid and slag detection
        fluid_segmentation_mask = self.fluid_head(refined_features)
        slag_segmentation_mask = self.slag_head(refined_features)
        
        # Prepare output dictionary
        model_outputs = {
            'fluid_mask': fluid_segmentation_mask,
            'slag_mask': slag_segmentation_mask,
            'quality': quality_prediction
        }
        
        # Add auxiliary outputs if available (used during training)
        if self.config['model']['aux_loss']:
            model_outputs.update(auxiliary_outputs)
            
        return model_outputs

# Training-specific loss functions and model creation utilities have been removed
# as they are not needed for inference. The EnhancedRefineNet class above contains
# everything needed for model inference and prediction.
