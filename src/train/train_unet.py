import os
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.optimizers import Optimizer, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.utils.generator import Parallel_array_reader_thread
from src.model.modified_unet import build_model


# Dataset
cwd = os.getcwd()
DATASET_PATH = os.path.join(cwd, "data", "dataset.hdf5")

# Model serialization params
SAVED_MODELS_FOLDER_PATH = os.path.join(cwd, "saved_models")
MODEL_NAME = "unet_4"

TRAIN_GENERATED_SAMPLES_FOLDER_PATH = os.path.join(cwd, SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "train_generated")
SAVED_MODEL_FOLDER_PATH = os.path.join(cwd, SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "model")

# Model training options
loss_fn = BinaryCrossentropy()
meanIoU_train = MeanIoU(2)

# Training hyperparameters
NUM_EPOCHS = 6
NUM_STEPS = 60000
BATCH_SIZE = 24

@tf.function
def train_step(model : Model, optim : Optimizer, X : np.ndarray, y : np.ndarray) -> tuple:
    with tf.GradientTape() as gtape:
        logits = model(X, training=True)
        loss = loss_fn(y_true=y, y_pred=logits)
    grads = gtape.gradient(loss, model.trainable_weights)
    optim.apply_gradients(zip(grads, model.trainable_weights))
    return loss, logits

def fit_unet(model : Model, optim : Optimizer) -> None:
    # Calculate y size to match model output
    model_output_shape = model.layers[-1].output_shape
    new_y_size = (model_output_shape[1], model_output_shape[2])

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} started!")

        loss_sum = 0
        log_step_counter = 1

        for step in range(NUM_STEPS):
            with Parallel_array_reader_thread(DATASET_PATH, BATCH_SIZE) as train_gen:
                x, y = next(train_gen)

            # Prepare training data
            y_resized = tf.image.resize(y, new_y_size)
            y_resized = tf.clip_by_value(y_resized, 0.0, 1.0)
            x_source, _ = x

            # Train model
            loss, logits = train_step(model, optim, x_source, y_resized)

            # Update loss
            loss_sum += loss

            # Calculate tensor for loss function
            iou_logits = tf.cast(logits + 0.5, dtype=tf.int32)
            iou_y_true = tf.cast(y_resized + 0.5, dtype=tf.int32)
            meanIoU_train.update_state(iou_y_true, iou_logits)

            if step % (NUM_STEPS / 20) == 0:
                # Calculate and print metrics/losses
                meanIoU_train_value = meanIoU_train.result()
                print(f"    {5*log_step_counter}%: loss = {loss_sum / (NUM_STEPS / 20):.4f}, meanIoU = {float(meanIoU_train_value):.4f}")
                meanIoU_train.reset_states()

                # Save examples of model predictions and true samples
                save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"true_{log_step_counter}.jpg"), y_resized[0])
                save_img(os.path.join(TRAIN_GENERATED_SAMPLES_FOLDER_PATH, f"sample_{log_step_counter}.jpg"), logits[0])

                # Update iterators
                loss_sum = 0
                log_step_counter += 1

                # Save model
                model.save(SAVED_MODEL_FOLDER_PATH)

if __name__ == "__main__":
    if not os.path.exists(TRAIN_GENERATED_SAMPLES_FOLDER_PATH):
        os.makedirs(TRAIN_GENERATED_SAMPLES_FOLDER_PATH)

    if os.path.exists(SAVED_MODEL_FOLDER_PATH):
        unet_model = keras.models.load_model(SAVED_MODEL_FOLDER_PATH)
        print(f"\nLoaded model from {SAVED_MODEL_FOLDER_PATH}")
    else:
        unet_model = build_model()
        print(f"\nBuilt new model!")

    optim = Adam()
    unet_model.compile(optimizer=optim)

    fit_unet(unet_model, optim)
