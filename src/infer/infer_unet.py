import imp
import os
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.generator import Parallel_array_reader_thread
from src.utils.image_processing import self_info_to_entropy_map, make_model_output_subplot
from src.utils.custom_loss_funcs import pred_to_self_information


cwd = os.getcwd()
INFER_COUNT = 5
DATASET_PATH = os.path.join(cwd, "data", "dataset.hdf5")
SAVED_MODELS_FOLDER_PATH = os.path.join(cwd, "saved_models")
MODEL_NAME = "s_unet_1"
RESULTS_FOLDER_PATH = os.path.join(cwd, "infered_images", MODEL_NAME)

def predict_to_subplot(model : Model, x_target : tf.Tensor, x_source : tf.Tensor, y : tf.Tensor) -> plt.figure:
    """Makes prediction and generating a subplot with pyplot. 

    Args:
        model (Model): model will be used for making prediction.
        x_target (tf.Tensor): target image input.
        x_source (tf.Tensor): source image input.
        y (tf.Tensor): segmentation map image.
    Returns:
        subplot_fig (plt.figure): subplot figure for prediction.
    """
    pred_source = model.predict(x_source)
    pred_target = model.predict(x_target)
    entropy_map = self_info_to_entropy_map(pred_to_self_information(pred_target))
    fig = make_model_output_subplot(x_target[0], x_source[0], y[0], pred_target[0, :, :, 1], pred_source[0, :, :, 1], entropy_map[0])
    return fig

def infer():
    if not os.path.exists(RESULTS_FOLDER_PATH):
        os.makedirs(RESULTS_FOLDER_PATH)

    model = keras.models.load_model(os.path.join(SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "segnet_model"))

    with Parallel_array_reader_thread(DATASET_PATH, 1) as train_gen:
        for i in range(INFER_COUNT):
            x, y = next(train_gen)
            x_s, x_t = x
            fig = predict_to_subplot(model, x_t, x_s, y)
            fig.savefig(os.path.join(RESULTS_FOLDER_PATH, f"subplot{i+1}.jpg"))
        print(f"Inference complete!")

if __name__ == "__main__":
    infer()
    