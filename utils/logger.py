"""
Logging utilities
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup logger with console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_model_summary(logger, model):
    """
    Log model summary to logger.
    
    Args:
        logger: Logger instance
        model: Keras model
    """
    logger.info("Model Architecture:")
    logger.info("=" * 60)
    
    # Capture model summary
    from io import StringIO
    import sys
    
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    model.summary()
    summary_string = sys.stdout.getvalue()
    
    sys.stdout = old_stdout
    
    for line in summary_string.split('\n'):
        if line.strip():
            logger.info(line)
    
    logger.info("=" * 60)


def log_config(logger, config):
    """
    Log experiment configuration.
    
    Args:
        logger: Logger instance
        config: ExperimentConfig object
    """
    logger.info("Experiment Configuration:")
    logger.info("=" * 60)
    logger.info(f"Experiment Name: {config.experiment_name}")
    logger.info(f"Random Seed: {config.random_seed}")
    logger.info("")
    logger.info("Data Configuration:")
    logger.info(f"  Train Images: {config.data.train_images_path}")
    logger.info(f"  Train Masks: {config.data.train_masks_path}")
    logger.info(f"  Number of Classes: {config.data.num_classes}")
    logger.info(f"  Image Size: {config.data.img_height}x{config.data.img_width}")
    logger.info(f"  Validation Split: {config.data.val_split}")
    logger.info(f"  Augmentation: {config.data.augmentation}")
    logger.info("")
    logger.info("Model Configuration:")
    logger.info(f"  Model Type: {config.model.model_type}")
    logger.info(f"  Backbone: {config.model.backbone}")
    logger.info(f"  Dropout Rate: {config.model.dropout_rate}")
    logger.info(f"  Batch Normalization: {config.model.use_batch_norm}")
    logger.info("")
    logger.info("Training Configuration:")
    logger.info(f"  Epochs: {config.training.epochs}")
    logger.info(f"  Batch Size: {config.training.batch_size}")
    logger.info(f"  Learning Rate: {config.training.learning_rate}")
    logger.info(f"  Optimizer: {config.training.optimizer}")
    logger.info(f"  Loss Function: {config.training.loss_function}")
    logger.info(f"  Early Stopping Patience: {config.training.early_stopping_patience}")
    logger.info("=" * 60)
