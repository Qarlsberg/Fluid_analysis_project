import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from typing import Dict, List

# Update quality readability mapping
QUALITY_READABILITY = {
    0: 1.0,    # Clear: fully readable
    1: 0.4,    # Steam: partially readable
    2: 0.7     # Fuzzy: poorly readable
}

class ResidualConv(nn.Module):
    """Memory-efficient residual convolution block"""
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
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        if self.skip is not None:
            residual = self.skip(residual)
        
        out += residual
        out = F.relu(out, inplace=True)
        return out

class QualityClassifier(nn.Module):
    """Memory-efficient quality classifier head"""
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
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)

class RefinementBlock(nn.Module):
    """Memory-efficient refinement block"""
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
        # Project high-level features if needed
        if self.high_proj is not None:
            high_feat = self.high_proj(high_feat)
        
        # Resize low-level feature to match high-level feature
        low_feat = F.interpolate(low_feat, size=high_feat.shape[2:], 
                               mode='bilinear', align_corners=False)
        
        # Refine low-level feature
        refined = self.residual(low_feat)
        
        # Apply channel attention
        channel_att = self.channel_gate(refined)
        refined = refined * channel_att
        
        # Apply spatial attention
        spatial_att = self.spatial_gate(refined)
        refined = refined * spatial_att
        
        # Combine features with residual connection
        out = high_feat + refined
        return out

class EnhancedRefineNet(nn.Module):
    """Memory-efficient RefineNet with quality classification"""
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
        # Store input size for resizing
        size = x.shape[2:]
        
        # Encoder path with gradient checkpointing if enabled
        x = self.layer0(x)     # 1/4
        x1 = self.layer1(x)    # 1/4
        x2 = self.layer2(x1)   # 1/8
        x3 = self.layer3(x2)   # 1/16
        x4 = self.layer4(x3)   # 1/32
        
        # Quality classification
        quality_pred = self.quality_classifier(x4)
        
        # Refinement path with deep supervision
        aux_outputs = {}
        if self.config['model']['aux_loss']:
            r4 = self.refine4(x4, x4)
            aux_outputs['aux4'] = F.interpolate(
                self.aux_head4(r4), size=size, mode='bilinear', align_corners=False
            )
            
            r3 = self.refine3(x3, r4)
            aux_outputs['aux3'] = F.interpolate(
                self.aux_head3(r3), size=size, mode='bilinear', align_corners=False
            )
            
            r2 = self.refine2(x2, r3)
            aux_outputs['aux2'] = F.interpolate(
                self.aux_head2(r2), size=size, mode='bilinear', align_corners=False
            )
            
            r1 = self.refine1(x1, r2)
        else:
            r4 = self.refine4(x4, x4)
            r3 = self.refine3(x3, r4)
            r2 = self.refine2(x2, r3)
            r1 = self.refine1(x1, r2)
        
        # Resize back to input resolution
        r1 = F.interpolate(r1, size=size, mode='bilinear', align_corners=False)
        
        # Output segmentation masks
        fluid_mask = self.fluid_head(r1)
        slag_mask = self.slag_head(r1)
        
        outputs = {
            'fluid_mask': fluid_mask,
            'slag_mask': slag_mask,
            'quality': quality_pred
        }
        
        if self.config['model']['aux_loss']:
            outputs.update(aux_outputs)
            
        return outputs

class EnhancedSegmentationLoss(nn.Module):
    """Combined BCE, Dice, auxiliary, and quality classification losses"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.aux_loss = config['model']['aux_loss']
        self.quality_weight = 0.2  # Weight for quality classification loss
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        # Get readability scores for quality-aware weighting
        readability = torch.tensor(
            [QUALITY_READABILITY[q.item()] for q in quality],
            device=pred.device
        ).view(-1, 1, 1)
        
        smooth = 1e-5
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        dice = 1 - ((2. * intersection + smooth) / 
                    (pred.sum(dim=1) + target.sum(dim=1) + smooth))
        
        # Weight dice loss by readability
        weighted_dice = dice * readability.squeeze()
        return weighted_dice.mean()
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, quality: torch.Tensor,
                  alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        # Get readability scores for quality-aware weighting
        readability = torch.tensor(
            [QUALITY_READABILITY[q.item()] for q in quality],
            device=pred.device
        ).view(-1, 1, 1)
        
        bce = -(target * torch.log(pred + 1e-7) + (1 - target) * torch.log(1 - pred + 1e-7))
        pt = torch.exp(-bce)
        focal_loss = alpha * (1-pt)**gamma * bce
        
        # Weight focal loss by readability
        weighted_focal = focal_loss * readability
        return weighted_focal.mean()
    
    def forward(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Main segmentation loss with quality awareness
        fluid_loss = 0.5 * self.focal_loss(pred['fluid_mask'], target['fluid'], target['quality']) + \
                    0.5 * self.dice_loss(pred['fluid_mask'], target['fluid'], target['quality'])
        slag_loss = 0.5 * self.focal_loss(pred['slag_mask'], target['slag'], target['quality']) + \
                   0.5 * self.dice_loss(pred['slag_mask'], target['slag'], target['quality'])
        
        main_loss = fluid_loss + slag_loss
        
        # Quality classification loss
        if 'quality' in target:
            quality_loss = F.cross_entropy(pred['quality'], target['quality'])
            main_loss = main_loss + self.quality_weight * quality_loss
        
        # Auxiliary losses with quality awareness
        if self.aux_loss and 'aux4' in pred:
            aux_weight = 0.4
            aux4_loss = self.focal_loss(pred['aux4'], target['fluid'], target['quality'])
            aux3_loss = self.focal_loss(pred['aux3'], target['fluid'], target['quality'])
            aux2_loss = self.focal_loss(pred['aux2'], target['fluid'], target['quality'])
            aux_loss = (aux4_loss + aux3_loss + aux2_loss) / 3
            return main_loss + aux_weight * aux_loss
        
        return main_loss

def create_model():
    """Create model and loss function"""
    with open('config.json', 'r') as f:
        import json
        config = json.load(f)
    
    model = EnhancedRefineNet(config)
    loss_fn = EnhancedSegmentationLoss(config)
    return model, loss_fn
