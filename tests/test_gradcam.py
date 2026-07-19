"""Unit tests for explainable AI Grad-CAM layer discovery."""

import pytest
import tensorflow as tf
import numpy as np
from src.models.efficientnet_model import build_model
from src.evaluation.explainability import GradCAMExplainer

def test_gradcam_find_last_conv():
    """Asserts that GradCAMExplainer locates the final Conv2D layer inside the model."""
    model = build_model(input_shape=(224, 224, 3), num_classes=4)
    explainer = GradCAMExplainer()
    
    layer, container = explainer._find_last_conv_layer(model)
    
    assert isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower()
    # Check it locates inside base model
    assert container.name.startswith("efficientnet")

def test_gradcam_heatmap_generation():
    """Asserts that explain outputs normalized grayscale heatmaps."""
    model = build_model(input_shape=(224, 224, 3), num_classes=4)
    explainer = GradCAMExplainer()
    
    # Create dummy preprocessed image tensor (1, H, W, 3)
    dummy_img = tf.ones((1, 224, 224, 3), dtype=tf.float32)
    
    heatmap = explainer.explain(model, dummy_img, class_idx=0)
    
    assert heatmap.shape == (7, 7) # EfficientNetB0 top_conv output size is 7x7 for 224x224 input
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0
