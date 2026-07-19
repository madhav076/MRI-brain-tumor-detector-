"""Grad-CAM Explainability Module.

Implements the Grad-CAM explainer inheriting from BaseExplainer, providing dynamic layer
discovery and heatmap overlay functions.
"""

import logging
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from src.evaluation.explainability.base_explainer import BaseExplainer

# Setup logging
logger = logging.getLogger(__name__)


class GradCAMExplainer(BaseExplainer):
    """Explainer class that generates Gradient-weighted Class Activation Maps (Grad-CAM)."""

    def __init__(self, last_conv_layer_name: str = None):
        """Initializes the GradCAMExplainer.

        Args:
            last_conv_layer_name (str): Optional name of last conv layer. If None, detected dynamically.
        """
        self.last_conv_layer_name = last_conv_layer_name

    def _find_last_conv_layer(
        self, model: tf.keras.Model
    ) -> Tuple[tf.keras.layers.Layer, tf.keras.Model]:
        """Dynamically scans the model to locate the final Conv2D layer.

        Args:
            model (tf.keras.Model): Main classification model.

        Returns:
            Tuple[tf.keras.layers.Layer, tf.keras.Model]: Last Conv2D layer and its containing model.
        """
        # If specific name is requested
        if self.last_conv_layer_name:
            try:
                # Check root model
                return model.get_layer(self.last_conv_layer_name), model
            except ValueError:
                # Check nested models
                for layer in model.layers:
                    if hasattr(layer, "layers"):
                        try:
                            return layer.get_layer(self.last_conv_layer_name), layer
                        except ValueError:
                            pass
                raise ValueError(f"Requested conv layer '{self.last_conv_layer_name}' not found.")

        # Otherwise, dynamically search for the last Conv2D layer in reverse order
        # Check nested backbones (e.g., efficientnetb0) first
        base_model = None
        for layer in model.layers:
            if layer.name.startswith("efficientnet"):
                base_model = layer
                break

        if base_model is not None:
            for layer in reversed(base_model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                    logger.info(
                        f"Dynamically discovered nested backbone conv layer: '{layer.name}'"
                    )
                    return layer, base_model

        # Fallback to scanning root model
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or "conv" in layer.name.lower():
                logger.info(f"Dynamically discovered root conv layer: '{layer.name}'")
                return layer, model

        raise ValueError("No Conv2D convolutional layer found in model structure.")

    def explain(self, model: tf.keras.Model, image: tf.Tensor, class_idx: int) -> np.ndarray:
        """Generates a Grad-CAM heatmap explaining the decision for a specific class.

        Uses a unified GradientTape approach that works for both flat and nested Keras
        Functional API models. In Keras Functional API (TF 2.10+), all sub-layer output
        tensors in a nested model are part of the top-level model's computation graph and
        are directly accessible for gradient tracking.

        Args:
            model (tf.keras.Model): Trained Keras model.
            image (tf.Tensor): Preprocessed input image tensor (H, W, C) or (1, H, W, C).
            class_idx (int): Output class target index.

        Returns:
            np.ndarray: Grayscale normalized heatmap (H, W) values in range [0, 1].
        """
        try:
            # Ensure 4D batch representation
            if len(image.shape) == 3:
                image = tf.expand_dims(image, axis=0)

            # Locate target convolutional layer
            target_conv_layer, container_model = self._find_last_conv_layer(model)

            # Determine whether target layer is nested inside a backbone
            is_nested = container_model is not model

            if is_nested:
                # For nested backbone (e.g., EfficientNetB0 as a sub-layer):
                # Build an intermediate model from the backbone's inputs to expose
                # the target conv layer output alongside the backbone's output.
                # Then manually apply the classification head layers to compute predictions.
                #
                # NOTE: The augmentation layer returns inputs unchanged at inference time
                # (training=False), so we can safely pass image directly to the backbone.
                backbone = container_model
                interim_model = tf.keras.Model(
                    inputs=backbone.inputs, outputs=[target_conv_layer.output, backbone.output]
                )

                # Identify the classification head layers (everything after the backbone)
                # We collect them in order after the backbone layer in the top-level model
                head_layers = []
                backbone_seen = False
                for layer in model.layers:
                    if layer.name == backbone.name:
                        backbone_seen = True
                        continue
                    # Skip augmentation (identity at inference) and input layer
                    if backbone_seen and layer.name != "mri_augmentation":
                        head_layers.append(layer)

                with tf.GradientTape() as tape:
                    # Run image through backbone; get conv features + backbone output
                    conv_outputs, backbone_out = interim_model(image, training=False)
                    # Watch the conv output tensor BEFORE computing downstream ops
                    tape.watch(conv_outputs)
                    # Pass backbone output through classification head
                    x = backbone_out
                    for layer in head_layers:
                        x = layer(x, training=False)
                    class_channel = x[:, class_idx]

                # Gradients of target class score w.r.t. conv feature map activations
                grads = tape.gradient(class_channel, conv_outputs)

            else:
                # For non-nested models: build a direct sub-model from model inputs
                # to [conv_output, predictions] — straightforward Functional API approach
                grad_model = tf.keras.Model(
                    inputs=model.inputs, outputs=[target_conv_layer.output, model.output]
                )
                with tf.GradientTape() as tape:
                    conv_outputs, preds = grad_model(image, training=False)
                    tape.watch(conv_outputs)
                    class_channel = preds[:, class_idx]

                grads = tape.gradient(class_channel, conv_outputs)

            # Validate gradients were computed
            if grads is None:
                raise ValueError(
                    "Grad-CAM gradient computation returned None. "
                    "The target conv layer may not be connected to the model output. "
                    f"Target layer: '{target_conv_layer.name}'"
                )

            # Calculate pooled gradients (mean intensity per filter channel)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            # Multiply feature map by pooled gradients
            conv_outputs_val = conv_outputs[0]
            heatmap = conv_outputs_val @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            # Apply ReLU (we only care about positive activations)
            heatmap = tf.maximum(heatmap, 0.0)

            # Normalize heatmap to [0, 1] range
            max_val = tf.reduce_max(heatmap)
            if max_val > 0:
                heatmap = heatmap / max_val

            return heatmap.numpy()

        except Exception as e:
            logger.error(f"Error generating Grad-CAM heatmap: {e}", exc_info=True)
            raise e

    @staticmethod
    def overlay_heatmap(
        image_path_or_array: Union[str, np.ndarray],
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Overlays the heatmap on top of the original grayscale MRI scan.

        Args:
            image_path_or_array (Union[str, np.ndarray]): File path or original image array.
            heatmap (np.ndarray): Heatmap matrix of values [0, 1].
            alpha (float): Blending opacity of heatmap overlay. Defaults to 0.4.
            colormap (int): OpenCV colormap conversion index. Defaults to COLORMAP_JET.

        Returns:
            np.ndarray: Color RGB overlaid image.
        """
        # Read or load image
        if isinstance(image_path_or_array, str):
            img = cv2.imread(image_path_or_array)
            if img is None:
                raise FileNotFoundError(f"Failed to read image at path: {image_path_or_array}")
        else:
            img = image_path_or_array.copy()

        # Re-scale to [0, 255] if float
        if img.dtype == np.float32 or img.dtype == np.float64:
            if np.max(img) <= 1.0:
                img = (img * 255.0).astype(np.uint8)

        # Handle grayscale to color RGB conversions
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif len(img.shape) == 3 and img.shape[2] == 3:
            # OpenCV loads color as BGR, ensure RGB representation
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Scale heatmap to [0, 255]
        heatmap_scaled = (heatmap * 255.0).astype(np.uint8)

        # Resize heatmap to match original image dimensions
        # cv2.resize expects (width, height) — use img.shape[1] for width, img.shape[0] for height
        heatmap_resized = cv2.resize(heatmap_scaled, (img.shape[1], img.shape[0]))

        # Apply OpenCV colormap representation
        heatmap_color = cv2.applyColorMap(heatmap_resized, colormap)
        # Convert BGR from applyColorMap to RGB
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Superimpose the heatmap on original image
        overlaid = cv2.addWeighted(heatmap_color, alpha, img, 1 - alpha, 0)
        return overlaid
