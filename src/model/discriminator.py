from tensorflow.keras import layers, Model


def build_model(input_shape : tuple) -> Model:
    """Builds dicriminator model. 

    Args:
        input_shape (tuple): shape of the input image.

    Returns:
        model (Model): keras model, not compiled.

    """
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(filters=16, kernel_size=4, strides=2, padding="valid")(inputs)
    c1_bn = layers.BatchNormalization()(c1)
    c1_a = layers.LeakyReLU()(c1_bn)

    c2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, padding="valid")(c1_a)
    c2_bn = layers.BatchNormalization()(c2)
    c2_a = layers.LeakyReLU()(c2_bn)

    c3 = layers.Conv2D(filters=64, kernel_size=4, strides=2, padding="valid")(c2_a)
    c3_bn = layers.BatchNormalization()(c3)
    c3_a = layers.LeakyReLU()(c3_bn)

    c4 = layers.Conv2D(filters=1, kernel_size=4, strides=2, padding="valid")(c3_a)
    c4_bn = layers.BatchNormalization()(c4)
    output = layers.LeakyReLU()(c4_bn)

    return Model(inputs, output)

