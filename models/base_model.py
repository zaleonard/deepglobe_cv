from abc import ABC, abstractmethod
import tensorflow as tf
from typing import Tuple, List, Optional

class BaseSegmentationModel(ABC):
    """base blass for segmentation models, enforces consistent interface and prevents instantiation of incomplete classes"""

    def __init__(
            self,
            input_shape: Tuple[int, int, int],
            num_classes: int,
            **kwargs
    ):
        """Args:
        input_shape: inpute image shape (H, W, C)   
        num_classes: number of segmentation classes
        **kwargs: Additional model-specific parameters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None  # Placeholder for the actual model instance
        self.kwargs = kwargs

    @abstractmethod
    def build_model(self) -> tf.keras.Model:
        """Build and return the segmentation model"""
        pass

    def compile_model(
            self,
            optimizer: str = 'adam',
            learning_rate: float = 1e-4,
            loss: str = 'categorical_crossentropy',
            metrics: Optional[List] = None
    ):
        """compile the model with given optimizer, loss and metrics
        Args:
            optimizer: optimizer name
            learning_rate: learning rate for the optimizer
            loss: loss function name
            metrics: list of metrics to evaluate during training
            
            Returns:
            Compiled Keras model
        """

        if self.model is None:
            self.model = self.build_model()

        #setup optimizer
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'sgd':
            opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer.lower == 'adamw':
            opt = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        #defaults
        if metrics is None:
            metrics = [
                'accuracy',
                  tf.keras.metrics.MeanIoU(num_classes=self.num_classes, name='mean_iou'),
                       tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')
                       ]
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)
        return self.model
    
    def summary(self):
        """print model summary"""
        if self.model is None:
            self.model = self.build_model()
        return self.model.summary()
    
    def save(self, filepath: str):
        """save model to filepath"""
        if self.model is None:
            raise ValueError("Model has not been built yet. Cannot save uninitialized model.  Call build_model() first.")
        self.model.save(filepath)

    def load_weights(self, filepath: str):
        """load model weights from filepath"""
        if self.model is None:
            self.model = self.build_model()
        self.model.load_weights(filepath)