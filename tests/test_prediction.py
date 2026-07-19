"""Unit tests for inference prediction pipelines."""

import numpy as np
import tensorflow as tf

from app.components.prediction_card import execute_inference
from src.models.efficientnet_model import build_model


def test_execute_inference():
    """Asserts that execute_inference compiles, executes, and yields prediction times."""
    # Build dummy model
    model = build_model(input_shape=(224, 224, 3), num_classes=4)

    # Create dummy uint8 grayscale image
    dummy_img = np.ones((256, 256), dtype=np.uint8) * 128

    # Execute inference
    probs, duration = execute_inference(dummy_img, model)

    assert probs.shape == (1, 4)
    assert duration > 0.0
    assert np.allclose(np.sum(probs), 1.0, atol=1e-5)
