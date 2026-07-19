"""Image preprocessing module.

Contains TensorFlow-based functions to resize, normalize, validate, and convert
MRI images, designed to run both on single images and batches.
"""

import logging
from typing import Tuple, Union
import numpy as np
import tensorflow as tf

# Set up logging
logger = logging.getLogger(__name__)


def validate_image(image: Union[np.ndarray, tf.Tensor]) -> bool:
    """Validates the structural and numerical integrity of an image.

    Checks for correct rank, non-empty dimensions, and occurrences of NaN or Inf values.

    Args:
        image (Union[np.ndarray, tf.Tensor]): Input image array or tensor.

    Returns:
        bool: True if image is valid, False otherwise.
    """
    try:
        # Convert to tensor for uniform checks
        tensor = tf.convert_to_tensor(image)

        # Check image rank
        rank = len(tensor.shape)
        if rank not in [2, 3, 4]:
            logger.warning(
                f"Invalid image rank: {rank}. Supported ranks are 2 (grayscale), 3 (HWC), or 4 (BHWC)."
            )
            return False

        # Check for empty dimensions
        if tf.reduce_any(tf.equal(tf.shape(tensor), 0)):
            logger.warning(f"Image contains zero dimensions: {tensor.shape}")
            return False

        # Check for NaN and Inf values only on floating point tensors
        if tensor.dtype.is_floating:
            if tf.reduce_any(tf.math.is_nan(tensor)):
                logger.warning("Image contains NaN values.")
                return False
            if tf.reduce_any(tf.math.is_inf(tensor)):
                logger.warning("Image contains Infinite values.")
                return False

        return True
    except Exception as e:
        logger.error(f"Error validating image: {e}")
        return False


def convert_to_rgb(image: tf.Tensor, is_bgr: bool = False) -> tf.Tensor:
    """Converts a grayscale, RGBA, BGR, or BGRA image array/tensor to a 3-channel RGB tensor.

    If the image has 4 channels, the alpha channel is automatically stripped.
    If the image is in BGR/BGRA format, channels are converted to RGB.
    If the image is grayscale, it is expanded to 3 channels.

    Args:
        image (tf.Tensor): Input image tensor of rank 2, 3, or 4.
        is_bgr (bool): If True, converts BGR/BGRA to RGB. Defaults to False.

    Returns:
        tf.Tensor: 3-channel RGB image tensor.
    """
    # Ensure tensor
    tensor = tf.convert_to_tensor(image)
    shape = tensor.shape
    rank = len(shape)

    if rank == 4:
        channels = shape[-1]
        # Drop alpha channel if 4 channels (RGBA/BGRA)
        if channels == 4:
            tensor = tensor[..., :3]
            channels = 3

        if channels == 3:
            if is_bgr:
                tensor = tensor[..., ::-1]  # BGR to RGB
            return tensor
        elif channels == 1:
            return tf.image.grayscale_to_rgb(tensor)
        else:
            raise ValueError(
                f"Unsupported number of channels in batch: {channels}. Expected 1, 3, or 4."
            )

    # Grayscale 2D image (H, W) -> expand to (H, W, 1) -> convert to RGB (H, W, 3)
    if rank == 2:
        tensor = tf.expand_dims(tensor, axis=-1)
        return tf.image.grayscale_to_rgb(tensor)

    if rank == 3:
        channels = shape[-1]
        # Drop alpha channel if 4 channels (RGBA/BGRA)
        if channels == 4:
            tensor = tensor[..., :3]
            channels = 3

        if channels == 3:
            if is_bgr:
                tensor = tensor[..., ::-1]  # BGR to RGB
            return tensor
        elif channels == 1:
            return tf.image.grayscale_to_rgb(tensor)
        else:
            raise ValueError(f"Unsupported number of channels: {channels}. Expected 1, 3, or 4.")

    raise ValueError(f"Unsupported image shape for RGB conversion: {shape}")


def resize_image(image: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    """Resizes the image to the specified target dimensions using bilinear interpolation.

    Args:
        image (tf.Tensor): Input image tensor.
        target_size (Tuple[int, int]): Target size as (height, width).

    Returns:
        tf.Tensor: Resized image tensor.
    """
    tensor = tf.convert_to_tensor(image)
    # tf.image.resize expects float type or returns float
    resized = tf.image.resize(tensor, size=target_size, method=tf.image.ResizeMethod.BILINEAR)
    return resized


def normalize_image(image: tf.Tensor, method: str = "minmax_01") -> tf.Tensor:
    """Normalizes the pixel intensities of the image tensor.

    Supported methods:
      - 'minmax_01': Scales pixel values from [0, 255] to [0, 1] by dividing by 255.0.
      - 'minmax_11': Scales pixel values from [0, 255] to [-1, 1] by (img / 127.5) - 1.0.
      - 'zscore': Normalizes to zero mean and unit variance.
      - 'dynamic_minmax': Scales the exact dynamic range of the image to [0, 1].

    Args:
        image (tf.Tensor): Input image tensor (cast to float).
        method (str): Normalization scheme. Defaults to 'minmax_01'.

    Returns:
        tf.Tensor: Normalized float32 image tensor.
    """
    tensor = tf.cast(image, dtype=tf.float32)
    epsilon = 1e-8

    if method == "minmax_01":
        return tensor / 255.0

    elif method == "minmax_11":
        return (tensor / 127.5) - 1.0

    elif method == "zscore":
        mean = tf.reduce_mean(tensor)
        std = tf.math.reduce_std(tensor)
        return (tensor - mean) / (std + epsilon)

    elif method == "dynamic_minmax":
        min_val = tf.reduce_min(tensor)
        max_val = tf.reduce_max(tensor)
        return (tensor - min_val) / (max_val - min_val + epsilon)

    else:
        logger.warning(f"Unknown normalization method: {method}. Defaulting to 'minmax_01'.")
        return tensor / 255.0


def preprocess_single_image(
    image: Union[np.ndarray, tf.Tensor],
    target_size: Tuple[int, int],
    normalize_method: str = "minmax_01",
    is_bgr: bool = False,
) -> tf.Tensor:
    """Applies the complete preprocessing pipeline to a single image.

    Validates, converts to RGB, resizes, and normalizes the image.

    Args:
        image (Union[np.ndarray, tf.Tensor]): Raw input image.
        target_size (Tuple[int, int]): Target size as (height, width).
        normalize_method (str): Method for pixel normalization. Defaults to 'minmax_01'.
        is_bgr (bool): If True, converts BGR/BGRA input to RGB. Defaults to False.

    Returns:
        tf.Tensor: Preprocessed 3D RGB image tensor (height, width, 3).
    """
    if not validate_image(image):
        raise ValueError("Image failed numerical or dimensional validation.")

    tensor = tf.convert_to_tensor(image)
    orig_shape = tuple(tensor.shape)

    tensor = convert_to_rgb(tensor, is_bgr=is_bgr)
    conv_shape = tuple(tensor.shape)

    tensor = resize_image(tensor, target_size)
    tensor = normalize_image(tensor, normalize_method)

    final_shape = (1, int(tensor.shape[0]), int(tensor.shape[1]), int(tensor.shape[2]))

    logger.info(f"Original: {orig_shape}")
    logger.info(f"Converted: {conv_shape}")
    logger.info(f"Input Tensor: {final_shape}")

    return tensor


def preprocess_batch(
    images: Union[np.ndarray, tf.Tensor],
    target_size: Tuple[int, int],
    normalize_method: str = "minmax_01",
    is_bgr: bool = False,
) -> tf.Tensor:
    """Preprocesses a batch of images.

    Args:
        images (Union[np.ndarray, tf.Tensor]): Input batch of images (NHWC, NHW, BHWC, etc).
        target_size (Tuple[int, int]): Target dimensions (height, width).
        normalize_method (str): Normalization scheme. Defaults to 'minmax_01'.
        is_bgr (bool): If True, converts BGR/BGRA input to RGB. Defaults to False.

    Returns:
        tf.Tensor: Preprocessed 4D RGB image tensor batch (batch_size, height, width, 3).
    """
    if not validate_image(images):
        raise ValueError("Image batch failed numerical or dimensional validation.")

    tensor = tf.convert_to_tensor(images)
    rank = len(tensor.shape)

    if rank != 4:
        raise ValueError(
            f"preprocess_batch expects a 4D tensor (Batch, Height, Width, Channels). Got rank {rank}."
        )

    orig_shape = tuple(tensor.shape)

    tensor = convert_to_rgb(tensor, is_bgr=is_bgr)
    conv_shape = tuple(tensor.shape)

    tensor = resize_image(tensor, target_size)
    tensor = normalize_image(tensor, normalize_method)

    final_shape = tuple(tensor.shape)

    logger.info(f"Original: {orig_shape}")
    logger.info(f"Converted: {conv_shape}")
    logger.info(f"Input Tensor: {final_shape}")

    return tensor
