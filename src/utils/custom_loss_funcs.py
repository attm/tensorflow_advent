from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import numpy as np


def tf_log2(value : tf.Tensor) -> tf.Tensor:
    """Calculates log2 using tf functions.

    Args:
        value (tf.Tensor): value for log2.
    Returns:

    """
    num = tf.math.log(tf.cast(value, dtype=tf.float32))
    denom = tf.math.log(tf.constant(2, dtype=num.dtype))
    return num / denom

def pred_to_self_information(prediction : tf.Tensor) -> tf.Tensor:
    """Converts probability maps made by model to self-information vector. 

    Args:
        prediction (tf.Tensor): softmax prediction from model.
    Returns:
        self_information (tf.Tensor): self-information vector, same shape as predictions.
    """
    channels = prediction.shape[3]

    self_information = -tf.math.multiply(prediction, tf_log2(prediction + 1e-30)) / tf_log2(channels)
    return self_information

def binary_ce_for_label(y_pred : tf.Tensor, y_label : float) -> tf.Tensor:
    """Calculates binary cross-entropy for given single label. Used for calculating loss for discriminator model that 
    uses self-information as input.

    Args:
        y_pred (tf.Tensor): probabilities prediction made by model.
        y_label (float): given label. New tensor filled with this value will be created.
    Returns:
        bce (tf.Tensor): binary crossentropy loss.
    """
    y_label_tensor = tf.fill(y_pred.shape, y_label)
    bce = BinaryCrossentropy()
    return bce(y_true=y_label_tensor, y_pred=y_pred)
