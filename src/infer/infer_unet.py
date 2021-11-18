import os
from tensorflow import keras
from tensorflow.keras.preprocessing.image import save_img
from src.utils.generator import Parallel_array_reader_thread

cwd = os.getcwd()
INFER_COUNT = 5
DATASET_PATH = os.path.join(cwd, "data", "dataset.hdf5")
SAVED_MODELS_FOLDER_PATH = os.path.join(cwd, "saved_models")
MODEL_NAME = "unet_4"
RESULTS_FOLDER_PATH = os.path.join(cwd, "infered_images", MODEL_NAME)

def infer():
    if not os.path.exists(RESULTS_FOLDER_PATH):
        os.makedirs(RESULTS_FOLDER_PATH)

    model = keras.models.load_model(os.path.join(SAVED_MODELS_FOLDER_PATH, MODEL_NAME, "model"))

    with Parallel_array_reader_thread(DATASET_PATH, 1) as train_gen:
        for i in range(INFER_COUNT):
            x, y = next(train_gen)
            x_s, x_t = x
            predicted_source = model.predict(x_s)
            predicted_target = model.predict(x_t)

            save_img(os.path.join(RESULTS_FOLDER_PATH, f"predicted_source_{i+1}.jpg"), predicted_source[0])
            save_img(os.path.join(RESULTS_FOLDER_PATH, f"predicted_target_{i+1}.jpg"), predicted_target[0])
            save_img(os.path.join(RESULTS_FOLDER_PATH, f"true_y_{i+1}.jpg"), y[0])

if __name__ == "__main__":
    infer()