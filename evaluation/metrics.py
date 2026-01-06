import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


class IoUMetric(tf.keras.metrics.Metric):
    """
    Intersection over Union (IoU) / Jaccard Index metric for segmentation.
    Measures overlap between predicted and ground truth regions.
    """
    
    def __init__(self, num_classes=4, name='iou', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_iou = self.add_weight(name='total_iou', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class indices
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        # Flatten
        y_true_flat = K.flatten(y_true_class)
        y_pred_flat = K.flatten(y_pred_class)
        
        # Calculate IoU for each class
        iou_sum = 0.0
        valid_classes = 0
        
        for class_id in range(self.num_classes):
            true_mask = tf.equal(y_true_flat, class_id)
            pred_mask = tf.equal(y_pred_flat, class_id)
            
            intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pred_mask), tf.float32))
            union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, pred_mask), tf.float32))
            
            # Avoid division by zero
            iou = tf.cond(
                union > 0,
                lambda: intersection / union,
                lambda: tf.constant(0.0)
            )
            
            iou_sum += iou
            valid_classes += 1
        
        mean_iou = iou_sum / tf.cast(valid_classes, tf.float32)
        
        self.total_iou.assign_add(mean_iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.total_iou / self.count
    
    def reset_state(self):
        self.total_iou.assign(0.0)
        self.count.assign(0.0)


class DiceMetric(tf.keras.metrics.Metric):
    """
    Dice coefficient metric (also known as F1 score for segmentation).
    Measures overlap between predicted and ground truth masks.
    """
    
    def __init__(self, num_classes=4, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.total_dice = self.add_weight(name='total_dice', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        smooth = 1e-6
        
        # Flatten
        y_true_flat = K.flatten(y_true)
        y_pred_flat = K.flatten(y_pred)
        
        # Calculate Dice coefficient
        intersection = K.sum(y_true_flat * y_pred_flat)
        dice = (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)
        
        self.total_dice.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.total_dice / self.count
    
    def reset_state(self):
        self.total_dice.assign(0.0)
        self.count.assign(0.0)


class PixelAccuracy(tf.keras.metrics.Metric):
    """
    Pixel-wise accuracy metric.
    Measures percentage of correctly classified pixels.
    """
    
    def __init__(self, name='pixel_accuracy', **kwargs):
        super(PixelAccuracy, self).__init__(name=name, **kwargs)
        self.total_correct = self.add_weight(name='total_correct', initializer='zeros')
        self.total_pixels = self.add_weight(name='total_pixels', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert to class indices
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        # Calculate correct predictions
        correct = tf.cast(tf.equal(y_true_class, y_pred_class), tf.float32)
        
        self.total_correct.assign_add(tf.reduce_sum(correct))
        self.total_pixels.assign_add(tf.cast(tf.size(y_true_class), tf.float32))
    
    def result(self):
        return self.total_correct / self.total_pixels
    
    def reset_state(self):
        self.total_correct.assign(0.0)
        self.total_pixels.assign(0.0)


class PerClassAccuracy(tf.keras.metrics.Metric):
    """
    Per-class accuracy metric.
    Reports accuracy for each class separately.
    """
    
    def __init__(self, num_classes=4, name='per_class_accuracy', **kwargs):
        super(PerClassAccuracy, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        
        # Create weight for each class
        self.class_correct = [self.add_weight(name=f'class_{i}_correct', initializer='zeros') 
                             for i in range(num_classes)]
        self.class_total = [self.add_weight(name=f'class_{i}_total', initializer='zeros')
                           for i in range(num_classes)]
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_class = tf.argmax(y_true, axis=-1)
        y_pred_class = tf.argmax(y_pred, axis=-1)
        
        for class_id in range(self.num_classes):
            mask = tf.equal(y_true_class, class_id)
            correct = tf.logical_and(mask, tf.equal(y_pred_class, class_id))
            
            self.class_correct[class_id].assign_add(tf.reduce_sum(tf.cast(correct, tf.float32)))
            self.class_total[class_id].assign_add(tf.reduce_sum(tf.cast(mask, tf.float32)))
    
    def result(self):
        # Return mean accuracy across classes
        accuracies = []
        for i in range(self.num_classes):
            acc = tf.cond(
                self.class_total[i] > 0,
                lambda i=i: self.class_correct[i] / self.class_total[i],
                lambda: tf.constant(0.0)
            )
            accuracies.append(acc)
        
        return tf.reduce_mean(accuracies)
    
    def reset_state(self):
        for i in range(self.num_classes):
            self.class_correct[i].assign(0.0)
            self.class_total[i].assign(0.0)


def calculate_confusion_matrix(y_true, y_pred, num_classes=4):
    """
    Calculate confusion matrix for multi-class segmentation.
    
    Args:
        y_true: Ground truth masks (one-hot encoded)
        y_pred: Predicted masks (softmax probabilities)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix as numpy array
    """
    y_true_class = np.argmax(y_true, axis=-1).flatten()
    y_pred_class = np.argmax(y_pred, axis=-1).flatten()
    
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            conf_matrix[true_class, pred_class] = np.sum(
                (y_true_class == true_class) & (y_pred_class == pred_class)
            )
    
    return conf_matrix
