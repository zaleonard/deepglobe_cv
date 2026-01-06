import argparse
import tensorflow as tf
from pathlib import Path
import sys

from config.base_config import ExperimentConfig, DataConfig, ModelConfig, TrainingConfig
from data.dataset import DeepGlobeDataset
from data.augmentation import get_default_augmentation
from models.custom.unet import UNetModel, AttentionUNetModel
from models.pretrained.efficientnet import EfficientNetSegmentation
from models.pretrained.mobilenet import MobileNetSegmentation
from training.trainer import ModelTrainer
from training.losses import get_loss_function
from utils.logger import setup_logger, log_config

def get_model_class(model_type: str):
    """Get model class based on model type"""
    models = {
        'unet': UNetModel,
        'attention_unet': AttentionUNetModel,
        'efficientnet': EfficientNetSegmentation,
        'mobilenet': MobileNetSegmentation,
    }
    return models.get(model_type.lower())

def main(args):
    tf.random.set_seed(args.seed)

    #create config
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        random_seed=args.seed,
        data=DataConfig(
            train_images_path=args.train_images,
            train_masks_path=args.train_masks,
            num_classes=args.num_classes,
            img_height=args.img_size,
            img_width=args.img_size,
            val_split=args.val_split,
            augmentation=args.augment
        ),
        model=ModelConfig(
            model_type=args.model_type,
            backbone=args.backbone,
            dropout_rate=args.dropout,
            use_batch_norm=args.batch_norm
        ),
        training=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer=args.optimizer,
            loss_function=args.loss
        )
    )

    #setup logger
    logger = setup_logger(args.experiment_name)
    logger.info("="*60)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model_type}")
    logger.info("="*60)
    
    # Log configuration
    log_config(logger, config)

    # Check if data files exist
    if not Path(args.train_images).exists():
        logger.error(f"Training images not found: {args.train_images}")
        logger.error("Please ensure training data is available.")
        sys.exit(1)
    
    if not Path(args.train_masks).exists():
        logger.error(f"Training masks not found: {args.train_masks}")
        logger.error("Please ensure training data is available.")
        sys.exit(1)

    # Load dataset
    logger.info("Loading training data...")
    try:
        dataset_obj = DeepGlobeDataset(
            images_path=args.train_images,
            masks_path=args.train_masks,
            num_classes=args.num_classes,
            augmentation=get_default_augmentation() if args.augment else None
        )
        
        logger.info(f"Total samples: {len(dataset_obj)}")
        
        # Split into train and validation
        train_dataset_obj, val_dataset_obj = dataset_obj.split_dataset(
            val_split=args.val_split,
            random_state=args.seed
        )
        
        logger.info(f"Training samples: {len(train_dataset_obj)}")
        logger.info(f"Validation samples: {len(val_dataset_obj)}")
        
        # Get class distribution
        class_dist = dataset_obj.get_class_distribution()
        logger.info("Class distribution:")
        for cls, count in class_dist.items():
            logger.info(f"  Class {cls}: {count} pixels")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create TensorFlow datasets
    train_dataset = train_dataset_obj.get_tf_dataset(
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataset = val_dataset_obj.get_tf_dataset(
        batch_size=args.batch_size,
        shuffle=False
    )

    # Build model
    logger.info(f"Building {args.model_type} model...")
    model_class = get_model_class(args.model_type)
    
    if model_class is None:
        logger.error(f"Unknown model type: {args.model_type}")
        logger.error("Available models: unet, attention_unet, efficientnet, mobilenet")
        sys.exit(1)
    
    try:
        model_builder = model_class(
            input_shape=(args.img_size, args.img_size, 3),
            num_classes=args.num_classes,
            backbone=args.backbone,
            dropout_rate=args.dropout,
            use_batch_norm=args.batch_norm
        )
        
        # Get loss function
        loss_fn = get_loss_function(args.loss)
        
        model = model_builder.compile_model(
            optimizer=args.optimizer,
            learning_rate=args.learning_rate,
            loss=loss_fn
        )
        
        logger.info("Model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")
        
    except Exception as e:
        logger.error(f"Error building model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Train model
    try:
        trainer = ModelTrainer(config)
        history = trainer.train(model, train_dataset, val_dataset)
        
        logger.info("="*60)
        logger.info("Training Summary:")
        logger.info(f"  Final Training Loss: {history['loss'][-1]:.4f}")
        logger.info(f"  Final Validation Loss: {history['val_loss'][-1]:.4f}")
        logger.info(f"  Best Validation Loss: {min(history['val_loss']):.4f}")
        logger.info("="*60)
        
        # Print TensorBoard command
        logger.info("\nView training progress with TensorBoard:")
        logger.info(f"  tensorboard --logdir={trainer.log_dir.parent}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train gating segmentation model")
        
        # Experiment settings
        parser.add_argument("--experiment-name", type=str, default="baseline",
                        help="Name of the experiment")
        parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
        
        # Data settings
        parser.add_argument("--train-images", type=str, default="training_images.pkl",
                        help="Path to training images")
        parser.add_argument("--train-masks", type=str, default="training_masks.pkl",
                        help="Path to training masks")
        parser.add_argument("--num-classes", type=int, default=7,
                        help="Number of classes (L, M, G, Background)")
        parser.add_argument("--img-size", type=int, default=256,
                        help="Input image size")
        parser.add_argument("--val-split", type=float, default=0.2,
                        help="Validation split ratio")
        parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation")
        
        # Model settings
        parser.add_argument("--model-type", type=str, default="unet",
                        choices=['unet', 'attention_unet', 'efficientnet', 'mobilenet'],
                        help="Model architecture")
        parser.add_argument("--backbone", type=str, default=None,
                        help="Backbone for transfer learning (e.g., efficientnetb0)")
        parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
        parser.add_argument("--batch-norm", action="store_true", default=True,
                        help="Use batch normalization")
        
        # Training settings
        parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
        parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
        parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate")
        parser.add_argument("--optimizer", type=str, default="adam",
                        choices=['adam', 'sgd', 'adamw'],
                        help="Optimizer")
        parser.add_argument("--loss", type=str, default="categorical_crossentropy",
                        choices=['categorical_crossentropy', 'dice', 'focal', 'combined'],
                        help="Loss function")
        
        args = parser.parse_args()
        main(args)



