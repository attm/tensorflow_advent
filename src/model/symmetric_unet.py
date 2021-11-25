"""Symmetrical unet model. 

Added BatchNormalization layers.
Removed some blocks, model is smaller than original.
Made all conv2d layers use padding="same", no crop layers needed anymore, only concatenate.
"""
from tensorflow.keras import layers, Model


def build_model(num_classes : int, input_shape : tuple, include_softmax : bool = True) -> Model:
    """Builds symmetrical u-net model. Height and Width of output will be same as input.

    Args:
        num_classes (int): number of classes, logits output will be image with number of channels = number of classes.
        input_shape (tuple): shape of the input image.
        include_softmax (bool): if True, then softmax layers will be included and model will return probabilities, 
                                if False, then output is last convolution layer and model will return logits.

    Returns:
        model (Model): keras model, not compiled.
    """
    inputs = layers.Input(shape=input_shape)

    ### ENCODER ###
    # Block 1
    e_b1_c1 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(inputs)
    e_b1_c1_bn = layers.BatchNormalization()(e_b1_c1)
    e_b1_c1_a = layers.Activation("relu")(e_b1_c1_bn)

    e_b1_c2 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(e_b1_c1_a)
    e_b1_c2_bn = layers.BatchNormalization()(e_b1_c2)
    e_b1_c2_a = layers.Activation("relu")(e_b1_c2_bn)

    # Block 2
    e_b2_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b1_c2_a)

    e_b2_c1 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(e_b2_maxpool)
    e_b2_c1_bn = layers.BatchNormalization()(e_b2_c1)
    e_b2_c1_a = layers.Activation("relu")(e_b2_c1_bn)

    e_b2_c2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(e_b2_c1_a)
    e_b2_c2_bn = layers.BatchNormalization()(e_b2_c2)
    e_b2_c2_a = layers.Activation("relu")(e_b2_c2_bn)

    # Block 3
    e_b3_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b2_c2_a)

    e_b3_c1 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(e_b3_maxpool)
    e_b3_c1_bn = layers.BatchNormalization()(e_b3_c1)
    e_b3_c1_a = layers.Activation("relu")(e_b3_c1_bn)

    e_b3_c2 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(e_b3_c1_a)
    e_b3_c2_bn = layers.BatchNormalization()(e_b3_c2)
    e_b3_c2_a = layers.Activation("relu")(e_b3_c2_bn)

    # Block 4 (Central block)
    e_b4_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b3_c2_a)

    e_b4_c1 = layers.Conv2D(filters=128, kernel_size=3, padding="same")(e_b4_maxpool)
    e_b4_c1_bn = layers.BatchNormalization()(e_b4_c1)
    e_b4_c1_a = layers.Activation("relu")(e_b4_c1_bn)

    e_b4_c2 = layers.Conv2D(filters=128, kernel_size=3, padding="same")(e_b4_c1_a)
    e_b4_c2_bn = layers.BatchNormalization()(e_b4_c2)
    e_b4_c2_a = layers.Activation("relu")(e_b4_c2_bn)

    ### DECODER ###
    # Block 3
    d_b3_ct1 = layers.Conv2DTranspose(filters=64, kernel_size=2, strides=(2, 2))(e_b4_c2_a)

    concat_b3 = layers.concatenate([d_b3_ct1, e_b3_c2_a], axis=-1)

    d_b3_c1 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(concat_b3)
    d_b3_c1_bn = layers.BatchNormalization()(d_b3_c1)
    d_b3_c1_a = layers.Activation("relu")(d_b3_c1_bn)

    d_b3_c2 = layers.Conv2D(filters=64, kernel_size=3, padding="same")(d_b3_c1_a)
    d_b3_c2_bn = layers.BatchNormalization()(d_b3_c2)
    d_b3_c2_a = layers.Activation("relu")(d_b3_c2_bn)

    # Block 2
    d_b2_ct1 = layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2, 2))(d_b3_c2_a)

    concat_b2 = layers.concatenate([d_b2_ct1, e_b2_c2_a], axis=-1)

    d_b2_c1 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(concat_b2)
    d_b2_c1_bn = layers.BatchNormalization()(d_b2_c1)
    d_b2_c1_a = layers.Activation("relu")(d_b2_c1_bn)

    d_b2_c2 = layers.Conv2D(filters=32, kernel_size=3, padding="same")(d_b2_c1_a)
    d_b2_c2_bn = layers.BatchNormalization()(d_b2_c2)
    d_b2_c2_a = layers.Activation("relu")(d_b2_c2_bn)

    # Block 1
    d_b1_ct1 = layers.Conv2DTranspose(filters=16, kernel_size=2, strides=(2, 2))(d_b2_c2_a)

    concat_b1 = layers.concatenate([d_b1_ct1, e_b1_c2_a], axis=-1)

    d_b1_c1 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(concat_b1)
    d_b1_c1_bn = layers.BatchNormalization()(d_b1_c1)
    d_b1_c1_a = layers.Activation("relu")(d_b1_c1_bn)

    d_b1_c2 = layers.Conv2D(filters=16, kernel_size=3, padding="same")(d_b1_c1_a)
    d_b1_c2_bn = layers.BatchNormalization()(d_b1_c2)
    d_b1_c2_a = layers.Activation("relu")(d_b1_c2_bn)

    output_c = layers.Conv2D(filters=num_classes, kernel_size=1, padding="same")(d_b1_c2_a)
    output_bn = layers.BatchNormalization()(output_c)

    if include_softmax:
        output = layers.Activation("softmax")(output_bn)
        return Model(inputs, output)
    else:
        output = layers.Activation("relu")(output_bn)
        return Model(inputs, output)
