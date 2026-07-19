"""Unit tests for EfficientNet model build compilation."""

import tensorflow as tf
from src.models.efficientnet_model import build_model
from src import config

def test_build_model():
    """Asserts model compile layout, output dimensions, and layers exist."""
    input_shape = (224, 224, 3)
    num_classes = 4
    
    model = build_model(input_shape=input_shape, num_classes=num_classes)
    
    # Assert type
    assert isinstance(model, tf.keras.Model)
    
    # Assert output shape
    assert model.output_shape == (None, num_classes)
    
    # Check mandatory layer names are present
    layer_names = [layer.name for layer in model.layers]
    assert "mri_augmentation" in layer_names
    assert "global_avg_pool" in layer_names
    assert "batch_norm" in layer_names
    assert "predictions" in layer_names
