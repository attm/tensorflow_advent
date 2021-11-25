import os
import pytest
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
from src.utils.custom_loss_funcs import pred_to_self_information, binary_ce_for_label


cwd = os.getcwd()
RESULTS_FOLDER_PATH = os.path.join(cwd, "src/tests", "test_results")

if not os.path.exists(RESULTS_FOLDER_PATH):
    os.makedirs(RESULTS_FOLDER_PATH)

def test_pred_to_self_information():
    test_image = tf.fill(dims=(1, 192, 192, 1), value=0.5)
    si = pred_to_self_information(test_image)
    assert si.shape == (1, 192, 192, 1)

def test_pred_to_self_information_complex():
    test_image_small = tf.fill(dims=(10, 192, 192, 3), value=0.5)
    si_small = pred_to_self_information(test_image_small)
    assert si_small.shape == (10, 192, 192, 3)

def test_pred_to_self_information_small_img():
    small_img_results_folder_path = os.path.join(RESULTS_FOLDER_PATH, "self_info_small_img")
    if not os.path.exists(small_img_results_folder_path):
        os.makedirs(small_img_results_folder_path)

    test_image = tf.fill(dims=(1, 32, 32, 1), value=0.5)
    si = pred_to_self_information(test_image)
    assert si.shape == (1, 32, 32, 1)
    save_img(os.path.join(small_img_results_folder_path, "img.jpg"), si[0])

def test_binary_ce_for_label():
    test_image = tf.fill(dims=(1, 192, 192, 1), value=0.5)
    bce_img = binary_ce_for_label(test_image, 1.0)
    assert float(bce_img) < 1.0 and float(bce_img) > 0.0
