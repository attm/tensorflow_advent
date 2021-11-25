import pytest
from src.model.symmetric_unet import build_model

def test_build_model():
    model = build_model(2, (192, 192, 3), True)
    assert model.layers[-1].output_shape == (None, 192, 192, 2)