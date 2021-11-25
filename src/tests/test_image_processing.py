import os
import pytest
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
from src.utils.image_processing import self_info_to_entropy_map, make_model_output_subplot


cwd = os.getcwd()
TEST_IMAGE_PATH = os.path.join(cwd, "src/tests/test_files", "test_pred.jpg")
RESULTS_FOLDER = os.path.join(cwd, "src/tests/test_results/image_processing")

def test_self_info_to_entropy_map():
    test_img = np.random.rand(1, 192, 192, 2)
    si_img = self_info_to_entropy_map(test_img)
    assert si_img.shape == (1, 192, 192, 1)
    assert si_img.min() >= 0.0
    assert si_img.max() <= 1.0

def test_make_model_output_subplot():
    test_img = img_to_array(load_img(TEST_IMAGE_PATH)) / 255.0
    img = make_model_output_subplot(test_img, 
                                    test_img,
                                    test_img,
                                    test_img,
                                    test_img,
                                    test_img)

    img.savefig(os.path.join(RESULTS_FOLDER, "test_model_output_subplot.jpg"))
