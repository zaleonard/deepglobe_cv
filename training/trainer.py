import tensorflow as tf
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import json

from config.base_config import ExperimentConfig
from data.dataset import DeepGlobeDataset
from utils.logger import setup_logger



class ModelTrainer:
    """Orchestrates model training with automatic experiemtn tracking.
    manages callbacks, logging, and model checkpointing"""
    def __init__(self, config:ExperimentConfig):
        self.config = config
        self.logger = setup_logger(config.experiment_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        #create dir
        self.experiment_dir = Path("experiments") / f"{config.experiment_name}_{timestamp}"
        self.experiment_dir.mkdir(parents=True, exists_ok=True)

        #create subdirs
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.log_dir = self.experiment_dir / "logs"
        self.results_dir = self.experiment_dir / "results"
        
        for directory in [self.checkpoint_dir, self.log_dir, self.results_dir]:
            directory.mkdir(exist_ok=True)

        config.save(self.experiment_dir / "config.yaml")

        self.logger.info(f"experiment directory: {self.experiment_dir}")
    
    def setup_callbacks(self) -> list:
        """setup training callbacks including Tensorboard, checkpoints, and early stopping.
        
        Returns:
        List of Keras callbacks
        """
        callbacks = []

        # TensorBoard
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(self.log_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch='10,20'
        )
        callbacks.append(tensorboard_callback)
        
        # Model checkpoint - save best model
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.checkpoint_dir / "best_model.keras"),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping_callback)
        
        # Reduce learning rate on plateau
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=self.config.training.reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr_callback)
        
        # CSV logger
        csv_callback = tf.keras.callbacks.CSVLogger(
            str(self.results_dir / "training_log.csv")
        )
        callbacks.append(csv_callback)
        
        return callbacks
    
    def train(self, model: tf.keras.Model, train_dataset: tf.data.Dataset, 
              val_dataset: tf.data.Dataset) -> Dict:
        """
        Train the model and save results.
        
        Args:
            model: Compiled Keras model
            train_dataset: Training dataset
            val_dataset: Validation dataset
        
        Returns:
            Dictionary containing training history
        """
        self.logger.info("="*60)
        self.logger.info("Starting training...")
        self.logger.info(f"Epochs: {self.config.training.epochs}")
        self.logger.info(f"Batch size: {self.config.training.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"Optimizer: {self.config.training.optimizer}")
        self.logger.info("="*60)
        
        # Setup callbacks
        callbacks = self.setup_callbacks()
        
        # Train model
        history = model.fit(
            train_dataset,
            epochs=self.config.training.epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model.save(str(self.checkpoint_dir / "final_model.keras"))
        
        # Save training history
        history_dict = history.history
        
        # Convert numpy types to Python types for JSON serialization
        history_json = {}
        for key, values in history_dict.items():
            history_json[key] = [float(v) for v in values]
        
        with open(self.results_dir / "history.json", 'w') as f:
            json.dump(history_json, f, indent=2)
        
        self.logger.info("="*60)
        self.logger.info("Training complete!")
        self.logger.info(f"Best model: {self.checkpoint_dir / 'best_model.keras'}")
        self.logger.info(f"Final model: {self.checkpoint_dir / 'final_model.keras'}")
        self.logger.info(f"Results: {self.results_dir}")
        self.logger.info("="*60)
        
        return history_dict
    
    def evaluate(self, model: tf.keras.Model, test_dataset: tf.data.Dataset) -> Dict:
        """
        Evaluate model on test dataset.
        
        Args:
            model: Trained Keras model
            test_dataset: Test dataset
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model on test set...")
        
        results = model.evaluate(test_dataset, verbose=1, return_dict=True)
        
        # Save results
        with open(self.results_dir / "test_results.json", 'w') as f:
            json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
        
        self.logger.info("Test results:")
        for metric, value in results.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        return results
