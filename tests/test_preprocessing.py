"""Unit tests for image validation, resizing, and normalization pipelines."""

import numpy as np
import tensorflow as tf
from src.data.preprocessing import (
    validate_image,
    convert_to_rgb,
    resize_image,
    normalize_image
)

def test_validate_image():
    """Verifies image validation passes for valid tensors and catches dimensions errors."""
    # Valid uint8 grayscale matrix
    valid_tensor = tf.ones((256, 256), dtype=tf.uint8)
    assert validate_image(valid_tensor) == True
    
    # Invalid empty tensor shape
    invalid_tensor = tf.zeros((0, 256), dtype=tf.uint8)
    assert validate_image(invalid_tensor) == False

def test_convert_to_rgb():
    """Asserts that 2D and 3D grayscale tensors are converted to 3-channel RGB."""
    gray_2d = tf.ones((100, 100), dtype=tf.float32)
    rgb = convert_to_rgb(gray_2d)
    assert rgb.shape == (100, 100, 3)

def test_resize_image():
    """Asserts image tensors dimensions are resized correctly."""
    img = tf.ones((100, 100, 3), dtype=tf.float32)
    resized = resize_image(img, (224, 224))
    assert resized.shape == (224, 224, 3)

def test_normalize_image():
    """Verifies that pixel values are normalized to target scales."""
    img = tf.constant([[127.5, 255.0], [0.0, 63.75]], dtype=tf.float32)
    
    # Minmax 0-1
    norm_01 = normalize_image(img, method="minmax_01")
    assert np.allclose(norm_01.numpy(), [[0.5, 1.0], [0.0, 0.25]])

def test_channel_conversions():
    """Asserts that various channel formats are converted correctly to 3-channel RGB."""
    # 1. RGBA (4 channels)
    rgba = tf.ones((10, 10, 4), dtype=tf.float32)
    rgb_from_rgba = convert_to_rgb(rgba, is_bgr=False)
    assert rgb_from_rgba.shape == (10, 10, 3)

    # 2. BGRA (4 channels, is_bgr=True)
    bgra = tf.constant([[[1.0, 2.0, 3.0, 4.0]]], dtype=tf.float32)  # B=1, G=2, R=3, A=4
    rgb_from_bgra = convert_to_rgb(bgra, is_bgr=True)
    assert rgb_from_bgra.shape == (1, 1, 3)
    # Check swapped: R=3, G=2, B=1
    assert np.allclose(rgb_from_bgra.numpy(), [[[3.0, 2.0, 1.0]]])

    # 3. Grayscale (1 channel)
    gray = tf.ones((10, 10, 1), dtype=tf.float32)
    rgb_from_gray = convert_to_rgb(gray)
    assert rgb_from_gray.shape == (10, 10, 3)
