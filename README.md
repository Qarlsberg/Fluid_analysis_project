# Fluid Analysis Project - Core Components

## Project Overview
This project implements an advanced fluid analysis system using deep learning. It's designed to:
- Detect and analyze fluid levels in real-time
- Identify slag presence
- Classify image quality (Clear, Steam, Fuzzy)
- Provide comprehensive visualization and analysis tools

## Changelog
### Version 1.0.0
- Initial implementation of EnhancedRefineNet architecture
- Real-time fluid level detection and analysis
- Quality classification system
- Streamlit-based inference application
- Memory-optimized training pipeline
- TensorBoard integration for monitoring

## Training Configuration

### Data Pipeline Settings
```json
{
    "cvat_annotations_path": "Cvat_dataset/annotations.xml",
    "images_dir": "Cvat_dataset/images/Train",
    "batch_size": 32,
    "image_size": [512, 512],
    "validation_split": 0.2,
    "num_workers": 4,
    "prefetch_factor": 2,
    "pin_memory": true
}
```

### Training Parameters
```json
{
    "learning_rate": 0.001,
    "epochs": 100,
    "early_stopping_patience": 15,
    "mixed_precision": true,
    "gradient_clip": 1.0,
    "empty_cache_freq": 10,
    "scheduler": {
        "type": "cosine",
        "warmup_epochs": 5,
        "min_lr": 1e-6
    }
}
```

### Data Augmentation
```json
{
    "enabled": true,
    "rotation_range": 5,
    "brightness_range": [0.9, 1.1],
    "contrast_range": [0.9, 1.1],
    "scale_range": [0.95, 1.05],
    "gaussian_noise": 0.2,
    "gaussian_blur": 0.1,
    "motion_blur": {
        "enabled": true,
        "kernel_size": [3, 5],
        "probability": 0.3
    },
    "perspective": {
        "enabled": true,
        "scale": 0.05,
        "probability": 0.3
    }
}
```

### Model Architecture
```json
{
    "backbone": "resnet34",
    "pretrained": true,
    "dropout": 0.2,
    "aux_loss": true,
    "memory_efficient": true,
    "channels": {
        "encoder": [64, 128, 256, 512],
        "decoder": [512, 256, 128, 64]
    }
}
```

### Debug Settings
```json
{
    "save_validation_predictions": true,
    "log_frequency": 100,
    "tensorboard": true,
    "save_memory_stats": true
}
```

## Usage
The project includes a Streamlit application for real-time analysis:

1. **Launch the App**
   ```bash
   streamlit run src/inference_app.py
   ```

2. **Interface Features**
   - Model selection from available checkpoints
   - ROI (Region of Interest) adjustment
   - Real-time visualization settings
   - Fluid level tracking and analysis
   - Quality classification with confidence scores

3. **Quality Classification**
   The system classifies image quality into three categories:
   - Clear (Readability: 1.0) - Optimal visibility
   - Steam (Readability: 0.4) - Partially obscured
   - Fuzzy (Readability: 0.7) - Reduced clarity

## 1. Enhanced RefineNet Architecture (`refine_net.py`)

### Key Components:

#### ResidualConv Block
```python
class ResidualConv(nn.Module):
    """Memory-efficient residual convolution block"""
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        # ... initialization code ...
```
- Implements residual connections for better gradient flow
- Uses batch normalization and dropout for regularization
- Memory-efficient design for handling high-resolution images

#### RefinementBlock
```python
class RefinementBlock(nn.Module):
    """Memory-efficient refinement block"""
    def __init__(self, high_channels: int, low_channels: int, out_channels: int, dropout: float = 0.2):
        # ... initialization code ...
```
- Combines high and low-level features effectively
- Implements channel and spatial attention mechanisms
- Optimized for memory efficiency

#### EnhancedRefineNet
```python
class EnhancedRefineNet(nn.Module):
    """Memory-efficient RefineNet with quality classification"""
```
- Main model architecture combining ResNet34 backbone with refinement blocks
- Multi-task learning: fluid detection, slag detection, and quality classification
- Deep supervision for better gradient flow
- Quality-aware feature processing

## 2. Training Pipeline (`train_model.py`)

### Key Features:

#### Memory Management
```python
def setup_cuda_memory():
    """Configure CUDA memory settings"""
    if torch.cuda.is_available():
        # Enable memory efficient features
        if torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
```
- Optimized CUDA memory settings
- Periodic memory clearing
- Support for mixed precision training

#### Advanced Training Features
```python
class Trainer:
    def __init__(self, config: Dict):
        # ... initialization code ...
```
- Mixed precision training support
- Cosine learning rate scheduling with warmup
- Early stopping and model checkpointing
- TensorBoard integration for monitoring
- Quality-aware loss functions

## 3. Inference Application (`inference_app.py`)

### Key Features:

#### Fluid Level Analysis
```python
def calculate_fluid_level(mask: np.ndarray) -> dict:
    """Calculate fluid level metrics including height and mass distribution"""
```
- Robust fluid level detection
- Outlier handling
- Mass distribution analysis

#### Real-time Processing
```python
class InferenceModel:
    def __init__(self, model_path: str, config_path: str):
        # ... initialization code ...
```
- Efficient model loading and inference
- Real-time mask visualization
- Quality classification with confidence scores

#### Interactive Visualization
```python
def overlay_masks(image: np.ndarray, predictions: dict, alpha: float = 0.5) -> np.ndarray:
    """Overlay prediction masks and quality info on image"""
```
- Real-time mask overlay
- Quality indicators
- Confidence visualization

## 4. Loss Functions

### Quality-Aware Segmentation Loss
```python
class EnhancedSegmentationLoss(nn.Module):
    """Combined BCE, Dice, auxiliary, and quality classification losses"""
```
- Quality-weighted loss computation
- Combines multiple loss terms:
  - Focal loss for handling class imbalance
  - Dice loss for better segmentation
  - Quality classification loss
  - Auxiliary losses for deep supervision

## Key Features and Benefits

1. **Memory Efficiency**
   - Optimized architecture for handling high-resolution images
   - Smart memory management with periodic clearing
   - Mixed precision training support

2. **Robust Training**
   - Quality-aware loss functions
   - Deep supervision
   - Advanced learning rate scheduling
   - Early stopping and checkpointing

3. **Real-time Processing**
   - Efficient inference pipeline
   - Real-time visualization
   - Quality classification with confidence scores

4. **Advanced Analysis**
   - Fluid level detection with outlier handling
   - Mass distribution analysis
   - Quality-aware processing

## Debugging Tools
For effective debugging and feedback:

1. **TensorBoard Integration**
   - Real-time monitoring of training metrics
   - Visualization of model architecture
   - Memory usage tracking
   - Sample predictions visualization

2. **Quality Metrics**
   - Per-frame quality classification
   - Readability scores
   - Confidence metrics
   - Mass distribution analysis

3. **Error Reporting**
   - Detailed error messages for model loading
   - Memory usage statistics
   - Performance metrics
   - Training/validation progress logs

This project implements a state-of-the-art fluid analysis system with robust training, efficient inference, and comprehensive visualization capabilities. The architecture is designed for memory efficiency and real-time processing, making it suitable for production environments.
l