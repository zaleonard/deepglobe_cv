"""
Custom loss functions for segmentation
"""
import tensorflow as tf
from tensorflow.keras import backend as K


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Dice coefficient for measuring overlap between predicted and ground truth masks.
    
    Args:
        y_true: Ground truth masks (one-hot encoded)
        y_pred: Predicted masks (softmax probabilities)
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient (higher is better)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred):
    """
    Dice loss = 1 - Dice coefficient
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
    
    Returns:
        Dice loss (lower is better)
    """
    return 1.0 - dice_coefficient(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for handling class imbalance.
    Focuses training on hard examples.
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Balancing factor
        gamma: Focusing parameter
    
    Returns:
        Focal loss
    """
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    # Calculate focal loss
    cross_entropy = -y_true * K.log(y_pred)
    focal_weight = alpha * K.pow(1 - y_pred, gamma)
    focal_loss = focal_weight * cross_entropy
    
    return K.mean(K.sum(focal_loss, axis=-1))


def combined_loss(y_true, y_pred, dice_weight=0.5, ce_weight=0.5):
    """
    Combined loss: Dice loss + Categorical Cross-Entropy
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        dice_weight: Weight for dice loss
        ce_weight: Weight for cross-entropy loss
    
    Returns:
        Combined loss
    """
    dl = dice_loss(y_true, y_pred)
    ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    return dice_weight * dl + ce_weight * K.mean(ce)


def weighted_categorical_crossentropy(class_weights):
    """
    Weighted categorical cross-entropy for class imbalance.
    
    Args:
        class_weights: Dictionary of class weights {class_id: weight}
    
    Returns:
        Weighted loss function
    """
    # Convert class weights to tensor
    weights = tf.constant([class_weights.get(i, 1.0) for i in range(len(class_weights))])
    
    def loss(y_true, y_pred):
        # Calculate weighted cross-entropy
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Weight each class
        weighted_ce = -y_true * K.log(y_pred) * weights
        
        return K.mean(K.sum(weighted_ce, axis=-1))
    
    return loss


def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    """
    Tversky loss - generalization of Dice loss.
    Better for imbalanced data.
    
    Args:
        y_true: Ground truth masks
        y_pred: Predicted masks
        alpha: Weight for false positives
        beta: Weight for false negatives
        smooth: Smoothing factor
    
    Returns:
        Tversky loss
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    true_pos = K.sum(y_true_f * y_pred_f)
    false_neg = K.sum(y_true_f * (1 - y_pred_f))
    false_pos = K.sum((1 - y_true_f) * y_pred_f)
    
    tversky_index = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    
    return 1.0 - tversky_index


# Dictionary of available losses
LOSS_FUNCTIONS = {
    'dice': dice_loss,
    'focal': focal_loss,
    'combined': combined_loss,
    'tversky': tversky_loss,
    'categorical_crossentropy': tf.keras.losses.categorical_crossentropy
}


def get_loss_function(loss_name: str, **kwargs):
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
    
    Returns:
        Loss function
    """
    if loss_name in LOSS_FUNCTIONS:
        loss_fn = LOSS_FUNCTIONS[loss_name]
        
        # Handle functions that need parameters
        if loss_name == 'combined':
            return lambda y_true, y_pred: combined_loss(y_true, y_pred, **kwargs)
        elif loss_name == 'focal':
            return lambda y_true, y_pred: focal_loss(y_true, y_pred, **kwargs)
        elif loss_name == 'tversky':
            return lambda y_true, y_pred: tversky_loss(y_true, y_pred, **kwargs)
        else:
            return loss_fn
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")