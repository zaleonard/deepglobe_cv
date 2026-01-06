import tensorflow as tf
from tensorflow.keras import layers, Model
from ..base_model import BaseSegmentationModel


class UNetModel(BaseSegmentationModel):
    def __init__(
            self, 
            input_shape=(256, 256, 3), 
            num_classes=7,
            filters_base=64,
            depth=4,
            dropout_rate=0.3,
            use_batch_norm=True,
            **kwargs):
        """
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            filters_base: Number of filters in first layer (doubles each level)
            depth: Depth of U-Net (number of downsampling steps)
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(input_shape, num_classes, **kwargs)
        self.filters_base = filters_base
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

    def conv_block(self, x, filters, kernel_size=3, use_bn=True):
        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def encoder_block(self, x, filters):
        #currently, conv block + max pooling
        conv = self.conv_block(x, filters, use_bn=self.use_batch_norm)
        pool = layers.MaxPooling2D(pool_size=(2,2))(conv)
        pool = layers.Dropout(self.dropout_rate)(pool)
        return conv, pool
    
    def decoder_block(self, x, skip_connection, filters):
        x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), 
                                   padding='same')(x)
        x = layers.concatenate([x, skip_connection])
        x = layers.Dropout(self.dropout_rate)(x)
        x = self.conv_block(x, filters, use_bn=self.use_batch_norm)
        return x
    
    def build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=self.input_shape)

        #encoder
        skip_connections = []
        x = inputs

        for i in range(self.depth):
            filters = self.filters_base * (2 ** i)
            skip, x = self.encoder_block(x, filters)
            skip_connections.append(skip)
        
        #bottleneck
        x = self.conv_block(x, self.filters_base * (2 ** self.depth), use_bn = self.use_batch_norm)

        #decoder
        for i in range(self.path -1, -1, -1):
            filters = self.filters_base * (2 ** i)
            x = self.decoder_block(x, skip_connections[i], filters)

        #output layer
        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(x)

        model = Model(inputs=inputs, outputs=outputs, name='unet')

        return model
    
class AttentionUNetModel(BaseSegmentationModel):
    """
    U-Net with attention gates for improved performance.
    Attention gates help the model focus on relevant features.
    """
    
    def __init__(self, input_shape=(256, 256, 1), num_classes=7,
                 filters_base=64, depth=4, dropout_rate=0.3,
                 use_batch_norm=True, **kwargs):
        """
        Args:
            input_shape: Input image shape (H, W, C)
            num_classes: Number of output classes
            filters_base: Number of filters in first layer
            depth: Depth of U-Net
            dropout_rate: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__(input_shape, num_classes, **kwargs)
        self.filters_base = filters_base
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
    
    def attention_gate(self, x, gating, inter_channels):
        """Attention gate mechanism"""
        theta_x = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), 
                               padding='same')(x)
        phi_g = layers.Conv2D(inter_channels, (1, 1), strides=(1, 1), 
                             padding='same')(gating)
        
        add = layers.add([theta_x, phi_g])
        act = layers.Activation('relu')(add)
        
        psi = layers.Conv2D(1, (1, 1), strides=(1, 1), padding='same')(act)
        psi = layers.Activation('sigmoid')(psi)
        
        # Upsample psi to match x shape if needed
        psi_upsampled = tf.image.resize(psi, tf.shape(x)[1:3])
        
        mul = layers.multiply([x, psi_upsampled])
        
        return mul
    
    def conv_block(self, x, filters, use_bn=True):
        x = layers.Conv2D(filters, 3, padding='same', 
                         kernel_initializer='he_normal')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = layers.Conv2D(filters, 3, padding='same',
                         kernel_initializer='he_normal')(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        return x
    
    def build_model(self) -> tf.keras.Model:
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder
        skip_connections = []
        x = inputs
        
        for i in range(self.depth):
            filters = self.filters_base * (2 ** i)
            x = self.conv_block(x, filters, use_bn=self.use_batch_norm)
            skip_connections.append(x)
            x = layers.MaxPooling2D(pool_size=(2, 2))(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Bottleneck
        x = self.conv_block(x, self.filters_base * (2 ** self.depth),
                           use_bn=self.use_batch_norm)
        
        # Decoder with attention
        for i in range(self.depth - 1, -1, -1):
            filters = self.filters_base * (2 ** i)
            x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2),
                                       padding='same')(x)
            
            # Apply attention gate
            skip_attention = self.attention_gate(skip_connections[i], x, filters // 2)
            
            x = layers.concatenate([x, skip_attention])
            x = layers.Dropout(self.dropout_rate)(x)
            x = self.conv_block(x, filters, use_bn=self.use_batch_norm)
        
        # Output
        outputs = layers.Conv2D(self.num_classes, (1, 1), 
                               activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='attention_unet')
        
        return model
    