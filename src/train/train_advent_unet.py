import imp
import os
import time
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, BinaryCrossentropy, Loss
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
import numpy as np
from src.utils.generator import Parallel_array_reader_thread
from src.model.symmetric_unet import build_model as build_segnet
from src.model.discriminator import build_model as build_discriminator
from src.utils.custom_loss_funcs import pred_to_self_information
from src.infer.infer_unet import predict_to_subplot


# Dataset
cwd = os.getcwd()
DATASET_PATH = os.path.join(cwd, "data", "dataset.hdf5")

# Segmentation model saving parameters
SAVED_MODELS_FOLDER_PATH = os.path.join(cwd, "saved_models")
MODEL_NAME = "s_unet_1"

TRAIN_GENERATED_SAMPLES_FOLDER_PATH = os.path.join(cwd, SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "train_generated")
SAVED_SEGNET_MODEL_FOLDER_PATH = os.path.join(cwd, SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "segnet_model")

# Image params
IMG_HEIGHT = 192
IMG_WIDTH = 192
IMG_CHANNELS = 1
NUM_CLASSES = 2

# Model training options
loss_fn = SparseCategoricalCrossentropy()

# Training hyperparameters
NUM_EPOCHS = 6
NUM_STEPS = 20000
BATCH_SIZE = 24

# Discriminator parameters
d_loss_fn = BinaryCrossentropy(from_logits=True)
D_SOURCE_LABEL = 0
D_TARGET_LABEL = 1

@tf.function
def train_step(model : Model, optim : Optimizer, loss_func : Loss, X : np.ndarray, y : np.ndarray) -> tuple:
    """Train step for model.

    Args:
        model (Model): keras model to be trained.
        optim (Optimizer) keras optimizer for model.
        loss_func (Loss): loss function for that model.
        X (np.ndarray): input for model.
        y (np.ndarray): ground-truth segmentation map.
    Return:
        loss (tf.Tensor): loss value for that step.
        logits (tf.Tensor): model output for that step (is it logits or probabilities depends on model, not on this function)
    """
    with tf.GradientTape() as gtape:
        logits = model(X, training=True)
        loss = loss_func(y_true=y, y_pred=logits)
        grads = gtape.gradient(loss, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))
        return loss, logits

@tf.function
def train_step_disc(model : Model, optim : Optimizer, loss_func : Loss, X : np.ndarray, y : np.ndarray) -> tuple:
    """Train step for discriminator model.

    Args:
        model (Model): keras model to be trained.
        optim (Optimizer) keras optimizer for model.
        loss_func (Loss): loss function for that model.
        X (np.ndarray): input for model.
        y (np.ndarray): ground-truth segmentation map.
    Return:
        loss (tf.Tensor): loss value for that step.
        logits (tf.Tensor): model output for that step (is it logits or probabilities depends on model, not on this function)
    """
    with tf.GradientTape() as gtape:
        logits = model(X, training=True)
        loss = loss_func(y_true=y, y_pred=logits)
        loss /= 2.0
        grads = gtape.gradient(loss, model.trainable_weights)
        optim.apply_gradients(zip(grads, model.trainable_weights))
        return loss, logits

@tf.function
def train_step_adv(model : Model, d_model : Model, d_optim : Optimizer, d_loss_fn : Loss, X : np.ndarray) -> tf.Tensor:
    """Train step for adversarial training for segmentation model and discriminator.

    Args:
        model (Model): segmentation model to be trained to fool the discriminator.
        d_model (Model): discriminator model.
        d_optim (Optimizer) keras optimizer for discriminator.
        d_loss_func (Loss): loss function for discriminator, segmentation will be trained based on that loss.
        X (np.ndarray): input for model.
    Returns:
        discriminator_loss (tf.Tensor): discriminator loss for this step.
        discriminator_output (tf.Tensor): discriminator logits output.
        logits_target (tf.Tensor): segmentation model output.
    """
    with tf.GradientTape() as gtape:
        logits_target = model(X, training=True)
        self_info_tensor = pred_to_self_information(logits_target)
        discriminator_output = d_model(self_info_tensor, training=False)
        disc_y_target = tf.fill(discriminator_output.shape, 1)
        discriminator_loss = d_loss_fn(y_pred=discriminator_output, y_true=disc_y_target)
        grads = gtape.gradient(discriminator_loss, discriminator_model.trainable_weights)
        d_optim.apply_gradients(zip(grads, discriminator_model.trainable_weights))
        return discriminator_loss, discriminator_output, logits_target

def fit_unet(model : Model, optim : Optimizer, discriminator_model : Model, discriminator_optim : Model) -> None:
    with Parallel_array_reader_thread(DATASET_PATH, BATCH_SIZE) as train_gen:
        for epoch in range(NUM_EPOCHS):
            print(f"===== Epoch {epoch+1}/{NUM_EPOCHS} started! =====")

            loss_sum = 0
            d_loss_sum = 0
            log_step_counter = 1
            step_time = time.time()

            for step in range(NUM_STEPS):
                x, y = next(train_gen)
                x_source, x_target = x

                # Prepare y image
                # Given y samples from dataset are already normalized to [0.0 - 1.0] range, but there are some outliners
                y = tf.clip_by_value(y, 0.0, 1.0)
                y = tf.cast(y + 0.5, dtype=tf.int32)

                # Disable discriminator model training
                for l in discriminator_model.layers:
                    l.trainable = False

                # Train segnet model
                loss, logits_source = train_step(model, optim, loss_fn, x_source, y)

                # Train segnet model to fool the discriminator using discriminator loss
                # Target images are used
                d_loss, d_ouput, logits_target = train_step_adv(model, discriminator_model, discriminator_optim, d_loss_fn, x_target)

                # Train discriminator using segnet model output for source and target inputs
                # Prepare labels for discriminator
                disc_y_source = tf.fill(d_ouput.shape, D_SOURCE_LABEL)
                disc_y_target = tf.fill(d_ouput.shape, D_TARGET_LABEL)

                # Enable discriminator model training
                for l in discriminator_model.layers:
                    l.trainable = True

                d_loss_source, _ = train_step_disc(discriminator_model, discriminator_optim, d_loss_fn, pred_to_self_information(logits_source), disc_y_source)
                d_loss_target, _ = train_step_disc(discriminator_model, discriminator_optim, d_loss_fn, pred_to_self_information(logits_target), disc_y_target)
                d_loss = d_loss_source + d_loss_target

                # Update loss
                loss_sum += loss
                d_loss_sum += d_loss

                if step % (NUM_STEPS / 20) == 0:
                    # Print losses
                    print(f"    {5*log_step_counter}% ({int(time.time() - step_time)}s): loss = {loss_sum / (NUM_STEPS / 20):.4f}, d_loss = {d_loss_sum / (NUM_STEPS / 20):.4f}")

                    # # Save examples of model predictions and true samples
                    # save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_x_source_c0.jpg"), x_source[0])
                    # save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_true_c0.jpg"), y[0, :, :, 0, None])
                    # save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_logits_c0.jpg"), logits_source[0, :, :, 0, None])
                    # #save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_true_c1.jpg"), y_resized[0, :, :, 1, None])
                    # save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_logits_c1.jpg"), logits_source[0, :, :, 1, None])

                    fig = predict_to_subplot(model, np.array(x_target), np.array(x_source), np.array(y))
                    fig.savefig(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_subplot.jpg"))

                    # Update iterators
                    loss_sum = 0
                    d_loss_sum = 0
                    log_step_counter += 1
                    step_time = time.time()

                    # Save model
                    model.save(SAVED_SEGNET_MODEL_FOLDER_PATH)

if __name__ == "__main__":
    if not os.path.exists(TRAIN_GENERATED_SAMPLES_FOLDER_PATH):
        os.makedirs(TRAIN_GENERATED_SAMPLES_FOLDER_PATH)

    # if os.path.exists(SAVED_SEGNET_MODEL_FOLDER_PATH):
    #     unet_model = keras.models.load_model(SAVED_SEGNET_MODEL_FOLDER_PATH)
    #     print(f"\nLoaded model from {SAVED_SEGNET_MODEL_FOLDER_PATH}")
    # else:
    #     unet_model = build_segnet(num_classes=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_softmax=True)
    #     print(f"\nBuilt new model!")

    unet_model = build_segnet(num_classes=2, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_softmax=True)

    optim = Adam(learning_rate=0.001)
    unet_model.compile(optimizer=optim)

    discriminator_model = build_discriminator(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES))
    d_optim = Adam(learning_rate=0.001)
    discriminator_model.compile(optimizer=d_optim)

    fit_unet(unet_model, optim, discriminator_model, d_optim)
