import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import json

from .metrics import calculate_confusion_matrix


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations and metrics.
    """
    
    def __init__(self, model: tf.keras.Model, num_classes: int = 4, 
                 class_names: List[str] = None):
        """
        Args:
            model: Trained Keras model
            num_classes: Number of classes
            class_names: List of class names (e.g., ['Background', 'L', 'M', 'G'])
        """
        self.model = model
        self.num_classes = num_classes
        self.class_names = class_names or [f'Class {i}' for i in range(num_classes)]
    
    def evaluate_dataset(self, dataset: tf.data.Dataset, save_dir: Path = None) -> Dict:
        """
        Evaluate model on a dataset and compute comprehensive metrics.
        
        Args:
            dataset: TensorFlow dataset
            save_dir: Directory to save results
        
        Returns:
            Dictionary of metrics
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all predictions and ground truth
        all_y_true = []
        all_y_pred = []
        
        print("Evaluating model...")
        for images, masks in dataset:
            predictions = self.model.predict(images, verbose=0)
            all_y_true.append(masks.numpy())
            all_y_pred.append(predictions)
        
        # Concatenate batches
        y_true = np.concatenate(all_y_true, axis=0)
        y_pred = np.concatenate(all_y_pred, axis=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        print("\nEvaluation Results:")
        print("=" * 50)
        for metric_name, value in metrics.items():
            if isinstance(value, dict):
                print(f"{metric_name}:")
                for k, v in value.items():
                    print(f"  {k}: {v:.4f}")
            else:
                print(f"{metric_name}: {value:.4f}")
        print("=" * 50)
        
        # Save results
        if save_dir:
            with open(save_dir / "evaluation_results.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Generate visualizations
            self.plot_confusion_matrix(y_true, y_pred, save_dir / "confusion_matrix.png")
            self.plot_class_performance(metrics, save_dir / "class_performance.png")
            self.visualize_predictions(y_true, y_pred, 
                                      save_path=save_dir / "predictions_sample.png",
                                      num_samples=5)
        
        return metrics
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: Ground truth masks
            y_pred: Predicted masks
        
        Returns:
            Dictionary of metrics
        """
        # Convert to class indices
        y_true_class = np.argmax(y_true, axis=-1)
        y_pred_class = np.argmax(y_pred, axis=-1)
        
        metrics = {}
        
        # Overall pixel accuracy
        correct_pixels = np.sum(y_true_class == y_pred_class)
        total_pixels = y_true_class.size
        metrics['pixel_accuracy'] = correct_pixels / total_pixels
        
        # Per-class metrics
        per_class_iou = {}
        per_class_dice = {}
        per_class_accuracy = {}
        
        for class_id in range(self.num_classes):
            class_name = self.class_names[class_id]
            
            # IoU
            true_mask = (y_true_class == class_id)
            pred_mask = (y_pred_class == class_id)
            
            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)
            
            iou = intersection / union if union > 0 else 0.0
            per_class_iou[class_name] = iou
            
            # Dice coefficient
            dice = (2 * intersection) / (np.sum(true_mask) + np.sum(pred_mask)) if (np.sum(true_mask) + np.sum(pred_mask)) > 0 else 0.0
            per_class_dice[class_name] = dice
            
            # Per-class accuracy
            class_correct = np.sum((y_true_class == class_id) & (y_pred_class == class_id))
            class_total = np.sum(y_true_class == class_id)
            
            accuracy = class_correct / class_total if class_total > 0 else 0.0
            per_class_accuracy[class_name] = accuracy
        
        metrics['per_class_iou'] = per_class_iou
        metrics['mean_iou'] = np.mean(list(per_class_iou.values()))
        metrics['per_class_dice'] = per_class_dice
        metrics['mean_dice'] = np.mean(list(per_class_dice.values()))
        metrics['per_class_accuracy'] = per_class_accuracy
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              save_path: Path = None):
        """Plot confusion matrix"""
        conf_matrix = calculate_confusion_matrix(y_true, y_pred, self.num_classes)
        
        # Normalize by row (true labels)
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix (Normalized)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_performance(self, metrics: Dict, save_path: Path = None):
        """Plot per-class performance metrics"""
        class_names = list(metrics['per_class_iou'].keys())
        iou_scores = list(metrics['per_class_iou'].values())
        dice_scores = list(metrics['per_class_dice'].values())
        accuracy_scores = list(metrics['per_class_accuracy'].values())
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, iou_scores, width, label='IoU', alpha=0.8)
        ax.bar(x, dice_scores, width, label='Dice', alpha=0.8)
        ax.bar(x + width, accuracy_scores, width, label='Accuracy', alpha=0.8)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def visualize_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                             images: np.ndarray = None, num_samples: int = 5,
                             save_path: Path = None):
        """Visualize sample predictions"""
        # Randomly select samples
        num_samples = min(num_samples, y_true.shape[0])
        indices = np.random.choice(y_true.shape[0], num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, idx in enumerate(indices):
            # Ground truth
            gt_mask = np.argmax(y_true[idx], axis=-1)
            axes[i, 0].imshow(gt_mask, cmap='tab10', vmin=0, vmax=self.num_classes-1)
            axes[i, 0].set_title(f'Sample {idx} - Ground Truth')
            axes[i, 0].axis('off')
            
            # Prediction
            pred_mask = np.argmax(y_pred[idx], axis=-1)
            axes[i, 1].imshow(pred_mask, cmap='tab10', vmin=0, vmax=self.num_classes-1)
            axes[i, 1].set_title(f'Sample {idx} - Prediction')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
