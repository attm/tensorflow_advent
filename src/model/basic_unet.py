"""Original U-Net realization made for image of shape (572, 572, 1).
Added BatchNormalization layers and use less filters. 
Output is (388, 388, 1).
"""
from tensorflow.keras import layers, Model


def build_model() -> Model:
    inputs = layers.Input(shape=(572, 572, 3))

    ### ENCODER ###
    # Block 1
    e_b1_c1 = layers.Conv2D(filters=16, kernel_size=3, padding="valid")(inputs)
    e_b1_c1_bn = layers.BatchNormalization()(e_b1_c1)
    e_b1_c1_a = layers.Activation("relu")(e_b1_c1_bn)

    e_b1_c2 = layers.Conv2D(filters=16, kernel_size=3, padding="valid")(e_b1_c1_a)
    e_b1_c2_bn = layers.BatchNormalization()(e_b1_c2)
    e_b1_c2_a = layers.Activation("relu")(e_b1_c2_bn)

    # Block 2
    e_b2_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b1_c2_a)

    e_b2_c1 = layers.Conv2D(filters=32, kernel_size=3, padding="valid")(e_b2_maxpool)
    e_b2_c1_bn = layers.BatchNormalization()(e_b2_c1)
    e_b2_c1_a = layers.Activation("relu")(e_b2_c1_bn)

    e_b2_c2 = layers.Conv2D(filters=32, kernel_size=3, padding="valid")(e_b2_c1_a)
    e_b2_c2_bn = layers.BatchNormalization()(e_b2_c2)
    e_b2_c2_a = layers.Activation("relu")(e_b2_c2_bn)

    # Block 3
    e_b3_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b2_c2_a)

    e_b3_c1 = layers.Conv2D(filters=64, kernel_size=3, padding="valid")(e_b3_maxpool)
    e_b3_c1_bn = layers.BatchNormalization()(e_b3_c1)
    e_b3_c1_a = layers.Activation("relu")(e_b3_c1_bn)

    e_b3_c2 = layers.Conv2D(filters=64, kernel_size=3, padding="valid")(e_b3_c1_a)
    e_b3_c2_bn = layers.BatchNormalization()(e_b3_c2)
    e_b3_c2_a = layers.Activation("relu")(e_b3_c2_bn)

    # Block 4
    e_b4_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b3_c2_a)

    e_b4_c1 = layers.Conv2D(filters=128, kernel_size=3, padding="valid")(e_b4_maxpool)
    e_b4_c1_bn = layers.BatchNormalization()(e_b4_c1)
    e_b4_c1_a = layers.Activation("relu")(e_b4_c1_bn)

    e_b4_c2 = layers.Conv2D(filters=128, kernel_size=3, padding="valid")(e_b4_c1_a)
    e_b4_c2_bn = layers.BatchNormalization()(e_b4_c2)
    e_b4_c2_a = layers.Activation("relu")(e_b4_c2_bn)

    # Block 5 (central block)
    e_b5_maxpool = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="same")(e_b4_c2_a)

    e_b5_c1 = layers.Conv2D(filters=256, kernel_size=3, padding="valid")(e_b5_maxpool)
    e_b5_c1_bn = layers.BatchNormalization()(e_b5_c1)
    e_b5_c1_a = layers.Activation("relu")(e_b5_c1_bn)

    e_b5_c2 = layers.Conv2D(filters=256, kernel_size=3, padding="valid")(e_b5_c1_a)
    e_b5_c2_bn = layers.BatchNormalization()(e_b5_c2)
    e_b5_c2_a = layers.Activation("relu")(e_b5_c2_bn)

    ### DECODER ###
    # Block 4
    d_b4_ct1 = layers.Conv2DTranspose(filters=128, kernel_size=2, strides=(2, 2))(e_b5_c2_a)

    crop_b4 = layers.Cropping2D(cropping=(4, 4))(e_b4_c2_a)
    concat_b4 = layers.concatenate([d_b4_ct1, crop_b4], axis=-1)

    d_b4_c1 = layers.Conv2D(filters=128, kernel_size=3, padding="valid")(concat_b4)
    d_b4_c1_bn = layers.BatchNormalization()(d_b4_c1)
    d_b4_c1_a = layers.Activation("relu")(d_b4_c1_bn)

    d_b4_c2 = layers.Conv2D(filters=128, kernel_size=3, padding="valid")(d_b4_c1_a)
    d_b4_c2_bn = layers.BatchNormalization()(d_b4_c2)
    d_b4_c2_a = layers.Activation("relu")(d_b4_c2_bn)

    # Block 3
    d_b3_ct1 = layers.Conv2DTranspose(filters=64, kernel_size=2, strides=(2, 2))(d_b4_c2_a)

    crop_b3 = layers.Cropping2D(cropping=(16, 16))(e_b3_c2_a)
    concat_b3 = layers.concatenate([d_b3_ct1, crop_b3], axis=-1)

    d_b3_c1 = layers.Conv2D(filters=64, kernel_size=3, padding="valid")(concat_b3)
    d_b3_c1_bn = layers.BatchNormalization()(d_b3_c1)
    d_b3_c1_a = layers.Activation("relu")(d_b3_c1_bn)

    d_b3_c2 = layers.Conv2D(filters=64, kernel_size=3, padding="valid")(d_b3_c1_a)
    d_b3_c2_bn = layers.BatchNormalization()(d_b3_c2)
    d_b3_c2_a = layers.Activation("relu")(d_b3_c2_bn)

    # Block 2
    d_b2_ct1 = layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2, 2))(d_b3_c2_a)

    crop_b2 = layers.Cropping2D(cropping=(40, 40))(e_b2_c2_a)
    concat_b2 = layers.concatenate([d_b2_ct1, crop_b2], axis=-1)

    d_b2_c1 = layers.Conv2D(filters=32, kernel_size=3, padding="valid")(concat_b2)
    d_b2_c1_bn = layers.BatchNormalization()(d_b2_c1)
    d_b2_c1_a = layers.Activation("relu")(d_b2_c1_bn)

    d_b2_c2 = layers.Conv2D(filters=32, kernel_size=3, padding="valid")(d_b2_c1_a)
    d_b2_c2_bn = layers.BatchNormalization()(d_b2_c2)
    d_b2_c2_a = layers.Activation("relu")(d_b2_c2_bn)

    # Block 1
    d_b1_ct1 = layers.Conv2DTranspose(filters=16, kernel_size=2, strides=(2, 2))(d_b2_c2_a)

    crop_b1 = layers.Cropping2D(cropping=(88, 88))(e_b1_c2_a)
    concat_b1 = layers.concatenate([d_b1_ct1, crop_b1], axis=-1)

    d_b1_c1 = layers.Conv2D(filters=16, kernel_size=3, padding="valid")(concat_b1)
    d_b1_c1_bn = layers.BatchNormalization()(d_b1_c1)
    d_b1_c1_a = layers.Activation("relu")(d_b1_c1_bn)

    d_b1_c2 = layers.Conv2D(filters=16, kernel_size=3, padding="valid")(d_b1_c1_a)
    d_b1_c2_bn = layers.BatchNormalization()(d_b1_c2)
    d_b1_c2_a = layers.Activation("relu")(d_b1_c2_bn)

    output_c = layers.Conv2D(filters=1, kernel_size=3, padding="same")(d_b1_c2_a)
    output_bn = layers.BatchNormalization()(output_c)
    output = layers.Activation("softmax")(output_bn)

    return Model(inputs, output)
