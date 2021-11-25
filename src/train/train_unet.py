import os
import time
from tensorflow import keras
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy, Loss
import tensorflow as tf
import numpy as np
from src.utils.generator import Parallel_array_reader_thread
from src.model.symmetric_unet import build_model as build_segnet
from src.infer.infer_unet import predict_to_subplot


# Dataset
cwd = os.getcwd()
DATASET_PATH = os.path.join(cwd, "data", "dataset.hdf5")

# Segmentation model saving parameters
SAVED_MODELS_FOLDER_PATH = os.path.join(cwd, "saved_models")
MODEL_NAME = "unet_1"

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
NUM_EPOCHS = 1
NUM_STEPS = 20000
BATCH_SIZE = 24

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

def fit_unet(model : Model, optim : Optimizer) -> None:
    with Parallel_array_reader_thread(DATASET_PATH, BATCH_SIZE) as train_gen:
        for epoch in range(NUM_EPOCHS):
            print(f"===== Epoch {epoch+1}/{NUM_EPOCHS} started! =====")

            loss_sum = 0
            log_step_counter = 1
            step_time = time.time()

            for step in range(NUM_STEPS):
                x, y = next(train_gen)
                x_source, x_target = x

                # Prepare y image
                # Given y samples from dataset are already normalized to [0.0 - 1.0] range, but there are some outliners
                y = tf.clip_by_value(y, 0.0, 1.0)
                y = tf.cast(y + 0.5, dtype=tf.int32)

                # Train segnet model
                loss, logits_source = train_step(model, optim, loss_fn, x_source, y)

                # Update loss
                loss_sum += loss

                if step % (NUM_STEPS / 20) == 0:
                    # Print losses
                    print(f"    {5*log_step_counter}% ({int(time.time() - step_time)}s): loss = {loss_sum / (NUM_STEPS / 20):.4f}")

                    # Save model prediction as subplot
                    fig = predict_to_subplot(model, np.array(x_target), np.array(x_source), np.array(y))
                    fig.savefig(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"{log_step_counter}_subplot.jpg"))

                    # Update iterators
                    loss_sum = 0
                    log_step_counter += 1
                    step_time = time.time()

                    # Save model
                    model.save(SAVED_SEGNET_MODEL_FOLDER_PATH)

if __name__ == "__main__":
    if not os.path.exists(TRAIN_GENERATED_SAMPLES_FOLDER_PATH):
        os.makedirs(TRAIN_GENERATED_SAMPLES_FOLDER_PATH)

    if os.path.exists(SAVED_SEGNET_MODEL_FOLDER_PATH):
        unet_model = keras.models.load_model(SAVED_SEGNET_MODEL_FOLDER_PATH)
        print(f"\nLoaded model from {SAVED_SEGNET_MODEL_FOLDER_PATH}")
    else:
        unet_model = build_segnet(num_classes=NUM_CLASSES, input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), include_softmax=True)
        print(f"\nBuilt new model!")

    optim = Adam(learning_rate=0.001)
    unet_model.compile(optimizer=optim)

    fit_unet(unet_model, optim)
