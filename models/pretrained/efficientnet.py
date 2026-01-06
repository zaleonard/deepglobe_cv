import tensorflow as tf
from tensorflow.keras import layers, Model
from ..base_model import BaseSegmentationModel

class EfficientNetSegmentation(BaseSegmentationModel):
    """
    Segmentation model using EfficientNet as backbone.
    Implements encoder-decoder architecture with skip connections.
    """

    def __init__(
            self,
            input_shape=(256, 256, 3), 
            num_classes=7,
            backbone='EfficientNetB0',
            pretrained_weights='imagenet',
            freeze_backbone=False,
            dropout_rate=0.3,
            **kwargs
    ):
        """
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            backbone: EfficientNet variant to use as backbone
            pretrained_weights: Weights for backbone ('imagenet' or path)
            freeze_backbone: Whether to freeze backbone during training
            dropout_rate: Dropout rate in decoder
        """
        super().__init__(input_shape, num_classes, **kwargs)
        self.backbone = backbone
        self.pretrained_weights = pretrained_weights
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate

        def get_backbone(self):
            """Load EfficientNet backbone"""
            backbone_map = {
            'efficientnetb0': tf.keras.applications.EfficientNetB0,
            'efficientnetb1': tf.keras.applications.EfficientNetB1,
            'efficientnetb2': tf.keras.applications.EfficientNetB2,
            'efficientnetb3': tf.keras.applications.EfficientNetB3,
            'efficientnetb4': tf.keras.applications.EfficientNetB4,
            'efficientnetb5': tf.keras.applications.EfficientNetB5,
            'efficientnetb6': tf.keras.applications.EfficientNetB6,
            'efficientnetb7': tf.keras.applications.EfficientNetB7,
        }

            backbone_class = backbone_map.get(self.backbone.lower())
            if backbone_class is None:
                raise ValueError(f"Unsupported backbone: {self.backbone}")
            
            #adjust input shape for pretrained models (3 channels for Imagenet)
            backbone_input_shape = (self.input_shape[0], self.input_shape[1], 3)

            weights = self.pretrained_weights if self.pretrained_weights else None

            backbone = backbone_class(
                include_top=False,
                weights=weights,
                input_shape=backbone_input_shape
            )

            return backbone
        
        def build_model(self) -> tf.keras.Model:
            inputs = layers.Input(shape=self.input_shape)
            
            if self.input_shape[-1] == 3:
                x = layers.Conv2D(3, (1, 1), padding='same', name='input_adapter')(inputs)
            else:
                x = inputs

            backbone = self.get_backbone()
            if self.freeze_backbone:
                backbone.trainable = False
            
            layer_names = []
            #extract features and get inetrmediate layer outputs for skip connections
            if len(layer_names) >= 4:
                skip_layers = [layer_names[i] for i in [0, len(layer_names)//3, 2*len(layer_names)//3, -1]]

            else:
                skip_layers = layer_names 

            #encoder features
            x=backbone(x)
            encoder_outputs = [backbone.get_layer(name).output for name in skip_layers]

            #decoder with upsampling
            img_size = DataConfig.img_height
            decoder_filters = [img_size, img_size // 2, img_size // 4, img_size // 8]

            for i, filters in enumerate(decoder_filters):
                x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                        padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)
                x = layers.Dropout(self.dropout_rate)(x)
                
                # Add skip connection if available
                if i < len(encoder_outputs):
                    skip = encoder_outputs[-(i+1)]
                    # Resize skip connection to match decoder size
                    skip_resized = tf.image.resize(skip, tf.shape(x)[1:3])
                    x = layers.concatenate([x, skip_resized])
                
                x = layers.Conv2D(filters, (3, 3), padding='same')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation('relu')(x)

            # Final upsampling to match input size
            x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            
            # Output layer
            outputs = layers.Conv2D(self.num_classes, (1, 1), 
                                activation='softmax')(x)
            
            # Ensure output size matches input
            outputs = tf.image.resize(outputs, self.input_shape[:2])
            
            model = Model(inputs=inputs, outputs=outputs, 
                        name=f'{self.backbone_name}_segmentation')
        
            return model