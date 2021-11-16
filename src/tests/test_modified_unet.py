import pytest
from src.model.modified_unet import build_model

def test_build_model():
    model = build_model()
    assert model.layers[-1].output_shape == (None, 132, 132, 1)