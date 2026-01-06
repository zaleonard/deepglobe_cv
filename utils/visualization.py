"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns


def plot_training_history(history: dict, save_path: Path = None):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Training history dictionary from model.fit()
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['loss'], label='Training Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy (or first available metric)
    metric_key = None
    for key in ['accuracy', 'mean_iou', 'dice']:
        if key in history:
            metric_key = key
            break
    
    if metric_key:
        axes[1].plot(history[metric_key], label=f'Training {metric_key.title()}', linewidth=2)
        val_metric_key = f'val_{metric_key}'
        if val_metric_key in history:
            axes[1].plot(history[val_metric_key], label=f'Validation {metric_key.title()}', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_key.title())
        axes[1].set_title(f'Training and Validation {metric_key.title()}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_predictions(images: np.ndarray, ground_truth: np.ndarray, 
                    predictions: np.ndarray, num_samples: int = 5,
                    class_names: list = None, save_path: Path = None):
    """
    Visualize predictions vs ground truth.
    
    Args:
        images: Input images
        ground_truth: Ground truth masks (one-hot encoded)
        predictions: Predicted masks (softmax probabilities)
        num_samples: Number of samples to display
        class_names: List of class names
        save_path: Path to save plot (optional)
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Input image
        if images[i].shape[-1] == 1:
            axes[i, 0].imshow(images[i, :, :, 0], cmap='gray')
        else:
            axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f'Input Image {i+1}')
        axes[i, 0].axis('off')
        
        # Ground truth
        gt_mask = np.argmax(ground_truth[i], axis=-1)
        im_gt = axes[i, 1].imshow(gt_mask, cmap='tab10')
        axes[i, 1].set_title(f'Ground Truth {i+1}')
        axes[i, 1].axis('off')
        
        # Prediction
        pred_mask = np.argmax(predictions[i], axis=-1)
        im_pred = axes[i, 2].imshow(pred_mask, cmap='tab10')
        axes[i, 2].set_title(f'Prediction {i+1}')
        axes[i, 2].axis('off')
    
    # Add colorbar with class names if provided
    if class_names:
        cbar = plt.colorbar(im_pred, ax=axes, orientation='horizontal', 
                           fraction=0.05, pad=0.05)
        cbar.set_ticks(range(len(class_names)))
        cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(experiment_results: dict, save_path: Path = None):
    """
    Compare metrics across multiple experiments.
    
    Args:
        experiment_results: Dictionary of {experiment_name: metrics_dict}
        save_path: Path to save plot
    """
    experiments = list(experiment_results.keys())
    
    # Extract metrics
    metrics_to_plot = ['mean_iou', 'mean_dice', 'pixel_accuracy']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(experiments))
    width = 0.25
    
    for i, metric in enumerate(metrics_to_plot):
        values = [experiment_results[exp].get(metric, 0) for exp in experiments]
        ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title(), alpha=0.8)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(experiments, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(masks: list, num_classes: int = 4, 
                            class_names: list = None, save_path: Path = None):
    """
    Plot class distribution in dataset.
    
    Args:
        masks: List of mask arrays
        num_classes: Number of classes
        class_names: List of class names
        save_path: Path to save plot
    """
    class_counts = {i: 0 for i in range(num_classes)}
    
    for mask in masks:
        unique, counts = np.unique(mask, return_counts=True)
        for cls, count in zip(unique, counts):
            if cls < num_classes:
                class_counts[int(cls)] += count
    
    # Create bar plot
    plt.figure(figsize=(10, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    if class_names:
        labels = class_names
    else:
        labels = [f'Class {i}' for i in classes]
    
    plt.bar(labels, counts, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xlabel('Class')
    plt.ylabel('Number of Pixels')
    plt.title('Class Distribution in Dataset')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add percentage labels on bars
    total = sum(counts)
    for i, (label, count) in enumerate(zip(labels, counts)):
        percentage = (count / total) * 100
        plt.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()
