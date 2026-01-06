import tensorflow as tf

def get_augmentation_pipeline(
        rotation_range: float = 0.2, flip_horizontal: bool = True, 
        flip_vertical: bool = True, brightness_range: float = 0.1, contrast_range: float = 0.1):
    """create data augmentation pipeline for image-mask pairs.
    
    Args: rotation_range: Range for random rotations ( in radians)
    flip_horizontal: whether to randomly flip horizontally
    flip_vertical: whether to randomly flip vertically
    brightness_range: Range for brightness adjustment
    contrast_range: Range for contrast adjustment
    
    returns:
    Augmentation function compatible with tf.data.Dataset.map()"""

    @tf.function
    def augment(image, mask):
        #concatenate image and mask for synchronized transformations
        combined = tf.concat([image, mask], axis=-1)

        #random rotation
        if rotation_range > 0:
            angle = tf.random.uniform([], -rotation_range, rotation_range)
            combined = tfa_rotate(combined, angle)

        #random horizontal flip
        if flip_horizontal:
            if tf.random.uniform([]) > 0.5:
                combined = tf.image.flip_left_right(combined)
        
        #random vertical flip
        if flip_vertical:
            if tf.random.uniform([]) > 0.5:
                combined = tf.image.flip_up_down(combined)

        if contrast_range > 0:
            image = tf.image.random_contrast(image, 1-contrast_range, 1+ contrast_range)

        #clip values
        image = tf.clip_by_value(image, 0.0, 0.1)

        return image, mask
    
    return augment

def tfa_rotate(image, angle):
    """rotate image by angle (in radians).
    simple rotation without tensorflow_addons dependency"""

    angle_deg = angle * 180.0 / 3.14159

    return image #implement proper rotation using tensorflow-addons

def simple_augmentation():
    """simple augmentation pipeline with only flips"""
    @tf.function
    def augment(image, mask):
        """Apply simple augmentation to image-mask pair"""
        
        # Stack image and mask for synchronized transformations
        combined = tf.concat([image, mask], axis=-1)
        
        # Random horizontal flip
        if tf.random.uniform([]) > 0.5:
            combined = tf.image.flip_left_right(combined)
        
        # Random vertical flip
        if tf.random.uniform([]) > 0.5:
            combined = tf.image.flip_up_down(combined)
        
        # Random 90-degree rotations
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)
        combined = tf.image.rot90(combined, k=k)
        
        # Split back
        image = combined[..., :1]
        mask = combined[..., 1:]
        
        # Brightness and contrast adjustments
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.9, 1.1)
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, mask
    
    return augment

def get_default_augmentation():
    return simple_augmentation
