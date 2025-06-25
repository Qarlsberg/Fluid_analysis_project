import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Dict, Tuple
import time
import json
import gc
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import torchvision
import warnings

from data_pipeline import create_data_pipeline
from refine_net import create_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress sklearn warnings about zero division
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)

def setup_cuda_memory():
    """Configure CUDA memory settings"""
    if torch.cuda.is_available():
        # Enable memory efficient features
        if torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Set memory allocator settings
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
        
        # Empty cache
        torch.cuda.empty_cache()
        gc.collect()

def calculate_metrics(outputs, targets):
    """Calculate accuracy metrics"""
    # Convert outputs to predictions
    pred_masks = {
        'fluid': (outputs['fluid_mask'] > 0.5).float(),
        'slag': (outputs['slag_mask'] > 0.5).float()
    }
    pred_quality = outputs['quality'].argmax(dim=1)
    
    # Calculate metrics
    metrics = {}
    
    # Quality classification accuracy
    quality_acc = accuracy_score(
        targets['quality'].cpu().numpy(),
        pred_quality.cpu().numpy()
    )
    metrics['quality_accuracy'] = quality_acc
    
    # Precision, recall, F1 for quality
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets['quality'].cpu().numpy(),
        pred_quality.cpu().numpy(),
        average='weighted',
        zero_division=0
    )
    metrics['quality_precision'] = precision
    metrics['quality_recall'] = recall
    metrics['quality_f1'] = f1
    
    # IoU for masks
    for mask_name, pred_mask in pred_masks.items():
        intersection = (pred_mask * targets[mask_name]).sum(dim=(1,2))
        union = (pred_mask + targets[mask_name]).gt(0).float().sum(dim=(1,2))
        iou = (intersection / (union + 1e-6)).mean()
        metrics[f'{mask_name}_iou'] = iou.item()
    
    return metrics

class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        setup_cuda_memory()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if self.device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("No GPU found, training on CPU will be slow!")
        
        # Create directories
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path('logs')
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.log_dir)
        logger.info(f"TensorBoard logs will be saved to {self.log_dir}")
        
        # Initialize components
        self._init_components()
        
    def _init_components(self):
        """Initialize model, optimizer, and data loaders"""
        # Create data loaders
        self.train_loader, self.val_loader = create_data_pipeline(self.config)
        logger.info(f"Dataset size - Train: {len(self.train_loader.dataset)}, Val: {len(self.val_loader.dataset)}")
        
        # Create model and loss function
        self.model, self.loss_fn = create_model()
        self.model = self.model.to(self.device)
        
        # Log model graph to TensorBoard
        try:
            sample_input = torch.randn(1, 3, *self.config['data_pipeline']['image_size']).to(self.device)
            # Temporarily wrap model to return only one output for graph logging
            class ModelWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                def forward(self, x):
                    return self.model(x)['fluid_mask']
            
            wrapped_model = ModelWrapper(self.model)
            self.writer.add_graph(wrapped_model, sample_input)
            logger.info("Successfully logged model graph to TensorBoard")
        except Exception as e:
            logger.warning(f"Failed to log model graph: {e}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
        
        if not self.config.get('validation_only', False):
            # Optimizer with learning rate schedule
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            # Cosine learning rate scheduler with warmup
            warmup_epochs = self.config['training']['scheduler']['warmup_epochs']
            total_epochs = self.config['training']['epochs']
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_epochs - warmup_epochs,
                eta_min=self.config['training']['scheduler']['min_lr']
            )
            
            # Mixed precision training
            self.use_amp = self.device.type == 'cuda' and self.config['training']['mixed_precision']
            if self.use_amp:
                self.scaler = GradScaler()
                logger.info("Using mixed precision training")
    
    def _clear_memory(self):
        """Clear GPU memory cache"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()
    
    def _log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log metrics to TensorBoard"""
        for name, value in metrics.items():
            self.writer.add_scalar(f'{prefix}/{name}', value, step)
    
    def _log_images(self, images, outputs, targets, step: int, max_images: int = 4):
        """Log images to TensorBoard"""
        try:
            # Log input images
            grid = torchvision.utils.make_grid(images[:max_images])
            self.writer.add_image('Input', grid, step)
            
            # Log predicted masks
            for name in ['fluid_mask', 'slag_mask']:
                if name in outputs:
                    pred_mask = (outputs[name] > 0.5).float()
                    grid = torchvision.utils.make_grid(pred_mask[:max_images])
                    self.writer.add_image(f'Pred_{name}', grid, step)
            
            # Log target masks
            for name in ['fluid', 'slag']:
                if name in targets:
                    grid = torchvision.utils.make_grid(targets[name][:max_images])
                    self.writer.add_image(f'Target_{name}', grid, step)
        except Exception as e:
            logger.warning(f"Failed to log images: {e}")
    
    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_times = []
        all_metrics = []
        empty_cache_freq = self.config['training'].get('empty_cache_freq', 20)
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, masks, quality_labels) in enumerate(pbar):
            start_time = time.time()
            
            # Clear memory periodically
            if batch_idx % empty_cache_freq == 0:
                self._clear_memory()
            
            images = images.to(self.device, non_blocking=True)
            masks = {k: v.to(self.device, non_blocking=True) for k, v in masks.items()}
            quality_labels = quality_labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with autocast(device_type='cuda'):
                    output = self.model(images)
                    loss = self.loss_fn(output, {**masks, 'quality': quality_labels})
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(images)
                loss = self.loss_fn(output, {**masks, 'quality': quality_labels})
                loss.backward()
                self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(output, {**masks, 'quality': quality_labels})
                all_metrics.append(metrics)
            
            # Update metrics
            total_loss += loss.item()
            batch_time = time.time() - start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            avg_time = sum(batch_times) / len(batch_times)
            metrics_str = f"loss: {avg_loss:.4f}"
            if len(all_metrics) > 0:
                avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
                metrics_str += f", acc: {avg_metrics['quality_accuracy']:.3f}"
            pbar.set_postfix_str(metrics_str)
            
            # Memory stats
            if self.config['debug']['save_memory_stats'] and batch_idx % 10 == 0:
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                self.writer.add_scalar('Memory/Allocated_GB', allocated, batch_idx)
                self.writer.add_scalar('Memory/Reserved_GB', reserved, batch_idx)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        return total_loss / len(self.train_loader), avg_metrics
    
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (images, masks, quality_labels) in enumerate(tqdm(self.val_loader, desc="Validation")):
                images = images.to(self.device, non_blocking=True)
                masks = {k: v.to(self.device, non_blocking=True) for k, v in masks.items()}
                quality_labels = quality_labels.to(self.device, non_blocking=True)
                
                # Forward pass
                with autocast(device_type='cuda', enabled=getattr(self, 'use_amp', False)):
                    output = self.model(images)
                    loss = self.loss_fn(output, {**masks, 'quality': quality_labels})
                
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = calculate_metrics(output, {**masks, 'quality': quality_labels})
                all_metrics.append(metrics)
                
                # Log sample predictions periodically
                if batch_idx == 0:  # Log first batch
                    self._log_images(images, output, masks, self.current_epoch if hasattr(self, 'current_epoch') else 0)
        
        # Calculate average metrics
        avg_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0].keys()}
        return total_loss / len(self.val_loader), avg_metrics
    
    def save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'config': self.config
        }
        
        if not self.config.get('validation_only', False):
            checkpoint.update({
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
            })
            if getattr(self, 'use_amp', False):
                checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def train(self, num_epochs: int = 100):
        """Train the model"""
        if self.config.get('validation_only', False):
            logger.info("Running validation only...")
            val_loss, metrics = self.validate()
            logger.info("\nValidation Results:")
            logger.info(f"Loss: {val_loss:.4f}")
            for name, value in metrics.items():
                logger.info(f"{name}: {value:.4f}")
            self._log_metrics(metrics, 0, 'Validation')
            self.save_checkpoint(0, val_loss, metrics)
            return
        
        best_val_loss = float('inf')
        patience = self.config['training']['early_stopping_patience']
        patience_counter = 0
        start_time = time.time()
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss, train_metrics = self.train_epoch()
            logger.info(f"Training loss: {train_loss:.4f}")
            for name, value in train_metrics.items():
                logger.info(f"Training {name}: {value:.4f}")
            
            # Log training metrics
            self._log_metrics({'loss': train_loss}, epoch, 'Train')
            self._log_metrics(train_metrics, epoch, 'Train')
            
            # Clear memory before validation
            self._clear_memory()
            
            # Validation
            val_loss, val_metrics = self.validate()
            logger.info(f"Validation loss: {val_loss:.4f}")
            for name, value in val_metrics.items():
                logger.info(f"Validation {name}: {value:.4f}")
            
            # Log validation metrics
            self._log_metrics({'loss': val_loss}, epoch, 'Validation')
            self._log_metrics(val_metrics, epoch, 'Validation')
            
            # Learning rate schedule
            current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f"Learning rate: {current_lr:.6f}")
            self.writer.add_scalar('Train/learning_rate', current_lr, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_metrics)
                logger.info(f"New best validation loss: {val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Update scheduler
            self.scheduler.step()
            
            # Log timing
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            logger.info(f"Epoch time: {epoch_time:.1f}s, Total time: {total_time/60:.1f}m")
            self.writer.add_scalar('Time/epoch_seconds', epoch_time, epoch)
            self.writer.add_scalar('Time/total_minutes', total_time/60, epoch)
            
            # Clear memory at end of epoch
            self._clear_memory()
        
        logger.info("\nTraining completed!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Total training time: {(time.time() - start_time)/60:.1f} minutes")
        
        # Close TensorBoard writer
        self.writer.close()

def main():
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train(num_epochs=config['training']['epochs'])

if __name__ == "__main__":
    main()
