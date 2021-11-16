import pytest
from src.model.basic_unet import build_model

def test_build_model():
    model = build_model()
    assert model.layers[-1].output_shape == (None, 388, 388, 1)