"""
Custom callbacks for training monitoring
"""
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


class VisualizationCallback(Callback):
    """
    Callback to visualize predictions during training.
    Saves sample predictions at end of each epoch.
    """
    
    def __init__(self, val_dataset, save_dir, num_samples=3):
        """
        Args:
            val_dataset: Validation dataset
            save_dir: Directory to save visualizations
            num_samples: Number of samples to visualize
        """
        super().__init__()
        self.val_dataset = val_dataset
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_samples = num_samples
        
        # Get sample data
        for images, masks in val_dataset.take(1):
            self.sample_images = images[:num_samples]
            self.sample_masks = masks[:num_samples]
            break
    
    def on_epoch_end(self, epoch, logs=None):
        """Visualize predictions at end of epoch"""
        predictions = self.model.predict(self.sample_images, verbose=0)
        
        fig, axes = plt.subplots(self.num_samples, 3, figsize=(12, 4*self.num_samples))
        
        if self.num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(self.num_samples):
            # Input image
            axes[i, 0].imshow(self.sample_images[i, :, :, 0], cmap='gray')
            axes[i, 0].set_title('Input')
            axes[i, 0].axis('off')
            
            # Ground truth
            gt_mask = np.argmax(self.sample_masks[i], axis=-1)
            axes[i, 1].imshow(gt_mask, cmap='tab10', vmin=0, vmax=3)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Prediction
            pred_mask = np.argmax(predictions[i], axis=-1)
            axes[i, 2].imshow(pred_mask, cmap='tab10', vmin=0, vmax=3)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'epoch_{epoch+1:03d}.png', dpi=100, bbox_inches='tight')
        plt.close()


class MetricsLoggerCallback(Callback):
    """
    Custom callback to log additional metrics during training.
    """
    
    def __init__(self, log_file):
        """
        Args:
            log_file: Path to log file
        """
        super().__init__()
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(self, logs=None):
        """Initialize log file"""
        with open(self.log_file, 'w') as f:
            f.write("epoch,loss,val_loss,learning_rate\n")
    
    def on_epoch_end(self, epoch, logs=None):
        """Log metrics at end of each epoch"""
        logs = logs or {}
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch+1},{logs.get('loss', 0):.6f},"
                   f"{logs.get('val_loss', 0):.6f},{lr:.8f}\n")


class GradientLoggerCallback(Callback):
    """
    Callback to log gradient statistics during training.
    Useful for detecting gradient vanishing/explosion.
    """
    
    def __init__(self, log_freq=10):
        """
        Args:
            log_freq: Log gradients every N batches
        """
        super().__init__()
        self.log_freq = log_freq
        self.batch_count = 0
    
    def on_train_batch_end(self, batch, logs=None):
        """Log gradient statistics"""
        self.batch_count += 1
        
        if self.batch_count % self.log_freq == 0:
            gradients = []
            for layer in self.model.layers:
                if hasattr(layer, 'kernel') and layer.trainable:
                    weights = layer.kernel
                    # Note: Getting gradients requires tape, this is simplified
                    # In practice, you'd need to integrate with training loop
                    pass