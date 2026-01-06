import tensorflow as tf
from tensorflow.keras import layers, Model
from ..base_model import BaseSegmentationModel
from config.base_config import DataConfig

class MobileNetSegmentation(BaseSegmentationModel):
    """
    Segmentation model using MobileNetV2 as backbone.
    Lightweight and efficient for deployment.
    """
    
    def __init__(self, input_shape=(256, 256, 1), num_classes=7,
                 backbone='mobilenetv2', pretrained_weights='imagenet',
                 freeze_backbone=False, dropout_rate=0.3, **kwargs):
        """
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            backbone: 'mobilenetv2' or 'mobilenetv3small/large'
            pretrained_weights: 'imagenet' or None
            freeze_backbone: Whether to freeze backbone weights
            dropout_rate: Dropout rate
        """
        super().__init__(input_shape, num_classes, **kwargs)
        self.backbone_name = backbone
        self.pretrained_weights = pretrained_weights
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout_rate
    
    def get_backbone(self):
        backbone_input_shape = (self.input_shape[0], self.input_shape[1], 3)
        weights = self.pretrained_weights if self.pretrained_weights else None

        if self.backbone_name.lower() == 'mobilenetv2':
             backbone = tf.keras.applications.MobileNetV2(
                 input_shape=backbone_input_shape,
                 include_top=False,
                 weights=weights
             )
        elif self.backbone_name.lower() == 'mobilenetv3small':
             backbone = tf.keras.applications.MobileNetV3Small(
                 input_shape=backbone_input_shape,
                 include_top=False,
                 weights=weights
             )
        elif self.backbone_name.lower() == 'mobilenetv3large':
             backbone = tf.keras.applications.MobileNetV3Large(
                 input_shape=backbone_input_shape,
                 include_top=False,
                 weights=weights
             )
        else:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}")
        
        return backbone
    
    def build_model(self) -> tf.keras.Model:
        """Build MobileNet-based segmentation model"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Convert single-channel to 3-channel if needed
        if self.input_shape[-1] == 1:
            x = layers.Conv2D(3, (1, 1), padding='same', 
                            name='input_adapter')(inputs)
        else:
            x = inputs
        
        # Get backbone
        backbone = self.get_backbone()
        
        # Freeze backbone if requested
        if self.freeze_backbone:
            backbone.trainable = False
        
        # Extract skip connection layers (typical for MobileNetV2)
        if 'v2' in self.backbone_name.lower():
            skip_layer_names = [
                'block_1_expand_relu',   # 128x128
                'block_3_expand_relu',   # 64x64
                'block_6_expand_relu',   # 32x32
                'block_13_expand_relu',  # 16x16
            ]
        else:
            # For MobileNetV3, adjust layer names accordingly
            skip_layer_names = []
        
        # Get encoder outputs
        skip_connections = []
        for layer_name in skip_layer_names:
            try:
                skip_connections.append(backbone.get_layer(layer_name).output)
            except:
                pass  # Skip if layer doesn't exist
        
        # Backbone output
        x = backbone(x)
        
        # Decoder with skip connections
        decoder_filters = [512, 256, 128, 64]
        
        for i, filters in enumerate(decoder_filters):
            # Upsample
            x = layers.Conv2DTranspose(filters, (3, 3), strides=(2, 2),
                                       padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(self.dropout_rate)(x)
            
            # Add skip connection if available
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # Match dimensions
                skip_resized = tf.image.resize(skip, tf.shape(x)[1:3])
                x = layers.concatenate([x, skip_resized])
            
            # Convolutional block
            x = layers.Conv2D(filters, (3, 3), padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
        
        # Final upsampling
        x = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), 
                                   padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        # Output layer
        outputs = layers.Conv2D(self.num_classes, (1, 1),
                               activation='softmax')(x)
        
        # Resize to match input
        outputs = tf.image.resize(outputs, self.input_shape[:2])
        
        model = Model(inputs=inputs, outputs=outputs,
                     name=f'{self.backbone_name}_segmentation')
        
        return model