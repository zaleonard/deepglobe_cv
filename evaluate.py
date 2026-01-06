import argparse
import tensorflow as tf
from pathlib import Path
import sys

from data.dataset import GatingDataset
from evaluation.evaluator import ModelEvaluator
from utils.io import load_model
from utils.logger import setup_logger


def main(args):
    """Main evaluation function"""
    
    # Setup logger
    logger = setup_logger('evaluation')
    logger.info("="*60)
    logger.info("Model Evaluation")
    logger.info("="*60)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    try:
        model = load_model(args.model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        sys.exit(1)
    
    # Load test dataset
    logger.info("Loading test data...")
    
    if not Path(args.test_images).exists():
        logger.error(f"Test images not found: {args.test_images}")
        sys.exit(1)
    
    if not Path(args.test_masks).exists():
        logger.error(f"Test masks not found: {args.test_masks}")
        sys.exit(1)
    
    try:
        test_dataset_obj = GatingDataset(
            images_path=args.test_images,
            masks_path=args.test_masks,
            num_classes=args.num_classes,
            augmentation=None  # No augmentation for evaluation
        )
        
        logger.info(f"Test samples: {len(test_dataset_obj)}")
        
        test_dataset = test_dataset_obj.get_tf_dataset(
            batch_size=args.batch_size,
            shuffle=False
        )
        
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        sys.exit(1)
    
    # Create evaluator
    class_names = args.class_names.split(',') if args.class_names else None
    evaluator = ModelEvaluator(
        model=model,
        num_classes=args.num_classes,
        class_names=class_names
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate model
    logger.info("Evaluating model...")
    try:
        metrics = evaluator.evaluate_dataset(
            dataset=test_dataset,
            save_dir=output_dir
        )
        
        logger.info("\nEvaluation complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("\nKey Metrics:")
        logger.info(f"  Mean IoU: {metrics['mean_iou']:.4f}")
        logger.info(f"  Mean Dice: {metrics['mean_dice']:.4f}")
        logger.info(f"  Pixel Accuracy: {metrics['pixel_accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    
    # Model settings
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model (.keras file)")
    
    # Data settings
    parser.add_argument("--test-images", type=str, default="test_images.pkl",
                       help="Path to test images")
    parser.add_argument("--test-masks", type=str, default="test_masks.pkl",
                       help="Path to test masks")
    parser.add_argument("--num-classes", type=int, default=4,
                       help="Number of classes")
    parser.add_argument("--class-names", type=str, default="Background,L,M,G",
                       help="Comma-separated class names")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for evaluation")
    
    # Output settings
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results")
    
    args = parser.parse_args()
    main(args)
