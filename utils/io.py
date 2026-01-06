import tensorflow as tf
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List


def save_predictions(predictions: np.ndarray, filenames: List[str], 
                    save_dir: Path, format: str = 'numpy'):
    """
    Save model predictions to disk.
    
    Args:
        predictions: Predicted masks
        filenames: List of corresponding filenames
        save_dir: Directory to save predictions
        format: Save format ('numpy', 'pickle', or 'both')
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if format in ['numpy', 'both']:
        # Save as individual .npy files
        for pred, filename in zip(predictions, filenames):
            pred_path = save_dir / f"{Path(filename).stem}_pred.npy"
            np.save(pred_path, pred)
    
    if format in ['pickle', 'both']:
        # Save all predictions in one pickle file
        pred_dict = {filename: pred for filename, pred in zip(filenames, predictions)}
        with open(save_dir / 'all_predictions.pkl', 'wb') as f:
            pickle.dump(pred_dict, f)
    
    print(f"Saved {len(predictions)} predictions to {save_dir}")


def load_predictions(save_dir: Path, format: str = 'pickle') -> Dict:
    """
    Load predictions from disk.
    
    Args:
        save_dir: Directory containing predictions
        format: Load format ('pickle' or 'numpy')
    
    Returns:
        Dictionary of {filename: prediction}
    """
    save_dir = Path(save_dir)
    
    if format == 'pickle':
        with open(save_dir / 'all_predictions.pkl', 'rb') as f:
            predictions = pickle.load(f)
    
    elif format == 'numpy':
        predictions = {}
        for pred_file in save_dir.glob('*_pred.npy'):
            filename = pred_file.stem.replace('_pred', '')
            predictions[filename] = np.load(pred_file)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return predictions


def load_model(model_path: str, custom_objects: Dict = None):
    """
    Load Keras model from file.
    
    Args:
        model_path: Path to saved model
        custom_objects: Dictionary of custom objects (losses, metrics, etc.)
    
    Returns:
        Loaded Keras model
    """
    from training.losses import LOSS_FUNCTIONS
    from evaluation.metrics import IoUMetric, DiceMetric, PixelAccuracy
    
    # Default custom objects
    default_custom_objects = {
        'dice_loss': LOSS_FUNCTIONS['dice'],
        'focal_loss': LOSS_FUNCTIONS['focal'],
        'combined_loss': LOSS_FUNCTIONS['combined'],
        'IoUMetric': IoUMetric,
        'DiceMetric': DiceMetric,
        'PixelAccuracy': PixelAccuracy
    }
    
    if custom_objects:
        default_custom_objects.update(custom_objects)
    
    model = tf.keras.models.load_model(model_path, custom_objects=default_custom_objects)
    
    print(f"Loaded model from {model_path}")
    
    return model


def save_experiment_summary(experiment_dir: Path, config, history: Dict, 
                           metrics: Dict = None):
    """
    Save comprehensive experiment summary.
    
    Args:
        experiment_dir: Experiment directory
        config: ExperimentConfig object
        history: Training history
        metrics: Evaluation metrics (optional)
    """
    import json
    from datetime import datetime
    
    summary = {
        'experiment_name': config.experiment_name,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'data': config.data.__dict__,
            'model': config.model.__dict__,
            'training': config.training.__dict__
        },
        'training_summary': {
            'total_epochs': len(history['loss']),
            'final_train_loss': float(history['loss'][-1]),
            'final_val_loss': float(history['val_loss'][-1]),
            'best_val_loss': float(min(history['val_loss'])),
        }
    }
    
    if metrics:
        summary['evaluation_metrics'] = metrics
    
    with open(experiment_dir / 'experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved experiment summary to {experiment_dir / 'experiment_summary.json'}")


def export_to_tflite(model: tf.keras.Model, save_path: Path, 
                     representative_dataset: callable = None):
    """
    Export model to TensorFlow Lite format for deployment.
    
    Args:
        model: Keras model to export
        save_path: Path to save .tflite file
        representative_dataset: Function returning representative data for quantization
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optional: quantization
    if representative_dataset:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
    
    tflite_model = converter.convert()
    
    with open(save_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Exported model to TFLite format: {save_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")