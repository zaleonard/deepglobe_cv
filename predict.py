import argparse
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys
import pickle

from utils.io import load_model, save_predictions
from utils.logger import setup_logger
from utils.visualization import plot_predictions


def main(args):
    """Main prediction function"""
    
    # Setup logger
    logger = setup_logger('prediction')
    logger.info("="*60)
    logger.info("Model Prediction")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    try:
        model = load_model(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load input data
    logger.info(f"Loading input data from {args.input_path}...")
    
    if not Path(args.input_path).exists():
        logger.error(f"Input file not found: {args.input_path}")
        sys.exit(1)
    
    try:
        with open(args.input_path, 'rb') as f:
            images = pickle.load(f)
        
        logger.info(f"Loaded {len(images)} images")
        
        # Prepare images for prediction
        if isinstance(images, list):
            # Convert list to array
            img_height, img_width = images[0].shape
            x_data = np.zeros((len(images), img_height, img_width, 1), dtype=np.float32)
            for i, img in enumerate(images):
                # Normalize
                img_normalized = img / (img.max() + 1e-8)
                x_data[i, :, :, 0] = img_normalized
        else:
            # Already an array
            x_data = images
            if x_data.ndim == 3:
                x_data = x_data[..., np.newaxis]
            # Normalize
            x_data = x_data / (x_data.max() + 1e-8)
        
        logger.info(f"Input shape: {x_data.shape}")
        
    except Exception as e:
        logger.error(f"Error loading input data: {e}")
        sys.exit(1)
    
    # Make predictions
    logger.info("Making predictions...")
    try:
        predictions = model.predict(x_data, batch_size=args.batch_size, verbose=1)
        logger.info(f"Predictions shape: {predictions.shape}")
        
        # Convert to class indices
        pred_masks = np.argmax(predictions, axis=-1)
        
        logger.info("Predictions complete!")
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving predictions to {output_dir}...")
    
    # Generate filenames
    filenames = [f"sample_{i:04d}" for i in range(len(predictions))]
    
    # Save as numpy arrays
    save_predictions(
        predictions=predictions,
        filenames=filenames,
        save_dir=output_dir,
        format='both'  # Save as both numpy and pickle
    )
    
    # Save visualizations
    if args.visualize:
        logger.info("Creating visualizations...")
        
        # Create visualizations (without ground truth)
        num_samples = min(args.num_samples, len(predictions))
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Input image
            axes[i, 0].imshow(x_data[i, :, :, 0], cmap='gray')
            axes[i, 0].set_title(f'Input Image {i+1}')
            axes[i, 0].axis('off')
            
            # Prediction
            axes[i, 1].imshow(pred_masks[i], cmap='tab10', vmin=0, vmax=args.num_classes-1)
            axes[i, 1].set_title(f'Prediction {i+1}')
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'predictions_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir / 'predictions_visualization.png'}")
    
    logger.info("="*60)
    logger.info("Prediction complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    
    # Model settings
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model (.keras file)")
    
    # Input/Output settings
    parser.add_argument("--input-path", type=str, required=True,
                       help="Path to input images (.pkl file)")
    parser.add_argument("--output-dir", type=str, default="predictions",
                       help="Directory to save predictions")
    
    # Prediction settings
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for prediction")
    parser.add_argument("--num-classes", type=int, default=4,
                       help="Number of classes")
    
    # Visualization settings
    parser.add_argument("--visualize", action="store_true",
                       help="Create visualizations of predictions")
    parser.add_argument("--num-samples", type=int, default=5,
                       help="Number of samples to visualize")
    
    args = parser.parse_args()
    main(args)
