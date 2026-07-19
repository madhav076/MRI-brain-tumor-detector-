"""Base Explainability Module.

Defines the abstract interface for all predictions explainers (e.g. Grad-CAM, Score-CAM).
"""

from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf


class BaseExplainer(ABC):
    """Abstract base class for computer vision explainability models."""

    @abstractmethod
    def explain(self, model: tf.keras.Model, image: tf.Tensor, class_idx: int) -> np.ndarray:
        """Generates a normalized heatmap explaining model decision for class_idx.

        Args:
            model (tf.keras.Model): Trained Keras model instance.
            image (tf.Tensor): Preprocessed input image tensor of shape (H, W, C) or (1, H, W, C).
            class_idx (int): The target classification category index.

        Returns:
            np.ndarray: Grayscale normalized heatmap of shape (H, W) values in range [0, 1].
        """
        pass
