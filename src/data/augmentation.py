"""MRI image augmentation module.

Provides a custom TensorFlow/Keras pipeline for applying clinical and physical
augmentations to Brain MRI images. Each operation is documented with its clinical relevance.
"""

import logging
import tensorflow as tf

# Set up logging
logger = logging.getLogger(__name__)

class RandomShear(tf.keras.layers.Layer):
    """Custom Keras layer to apply random shear mapping using TensorFlow operations.

    Shear is useful for MRI scans to simulate minor perspective distortions,
    spatial warping from scanner gradients, or deviations in scan slice angle.
    """

    def __init__(self, shear_factor: float = 0.1, **kwargs):
        """Initializes the RandomShear layer.

        Args:
            shear_factor (float): Maximum shear range (horizontal and vertical).
        """
        super().__init__(**kwargs)
        self.shear_factor = shear_factor

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Applies random shear to input images.

        Args:
            inputs (tf.Tensor): Input batch of images (NHWC).
            training (bool): Whether the layer is in training mode.

        Returns:
            tf.Tensor: Sheared batch of images.
        """
        if not training or self.shear_factor == 0.0:
            return inputs

        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        # Generate random shear factors
        shx = tf.random.uniform([batch_size], -self.shear_factor, self.shear_factor)
        shy = tf.random.uniform([batch_size], -self.shear_factor, self.shear_factor)

        # Construct transformation matrices (inverse mapping for projective transform)
        # Affine shear matrix:
        # [ 1   shx  -shx*(w/2) ]
        # [ shy  1   -shy*(h/2) ]
        # [ 0    0    1         ]
        # 8-element flat representation: [a0, a1, a2, b0, b1, b2, c0, c1]
        a0 = tf.ones([batch_size])
        a1 = shx
        a2 = -shx * tf.cast(width, tf.float32) * 0.5
        b0 = shy
        b1 = tf.ones([batch_size])
        b2 = -shy * tf.cast(height, tf.float32) * 0.5
        c0 = tf.zeros([batch_size])
        c1 = tf.zeros([batch_size])

        transforms = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1], axis=1)

        # Apply transformation using the stable V3 raw op (available in TF 2.10+)
        # V3 adds explicit fill_value support for CONSTANT fill_mode
        inputs_float = tf.cast(inputs, tf.float32)
        output_shape = tf.stack([height, width])

        # Try V3 first (TF 2.10+), fallback to V2 for older TF versions
        try:
            sheared = tf.raw_ops.ImageProjectiveTransformV3(
                images=inputs_float,
                transforms=transforms,
                output_shape=output_shape,
                interpolation="BILINEAR",
                fill_mode="CONSTANT",
                fill_value=0.0
            )
        except AttributeError:
            # Fallback for environments where only V2 is available
            sheared = tf.raw_ops.ImageProjectiveTransformV2(
                images=inputs_float,
                transforms=transforms,
                output_shape=output_shape,
                interpolation="BILINEAR",
                fill_mode="CONSTANT"
            )

        return tf.cast(sheared, inputs.dtype)

    def get_config(self):
        config = super().get_config()
        config.update({"shear_factor": self.shear_factor})
        return config

    @classmethod
    def from_config(cls, config):
        """Explicitly reconstruct from serialized config dict."""
        return cls(**config)


class MRIAugmentationPipeline(tf.keras.layers.Layer):
    """Integrated Keras Augmentation pipeline for Brain MRI image classification.

    Applies rotation, translations, flips, zooms, brightness, and shear
    only during the training phase.
    """

    def __init__(
        self,
        image_size: tuple = (224, 224),
        rotation_range: float = 0.15,
        zoom_range: float = 0.1,
        shift_range: float = 0.1,
        brightness_range: float = 0.15,
        contrast_range: float = 0.15,
        shear_range: float = 0.1,
        **kwargs
    ):
        """Initializes the MRI Augmentation Pipeline.

        Args:
            image_size (tuple): Target image size (height, width).
            rotation_range (float): Max rotation angle in radians.
            zoom_range (float): Maximum zoom scaling in/out.
            shift_range (float): Max translation shift horizontally and vertically.
            brightness_range (float): Max brightness modification factor.
            contrast_range (float): Max contrast modification factor.
            shear_range (float): Max horizontal/vertical shear mapping.
        """
        super().__init__(**kwargs)
        self.image_size = image_size

        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.shift_range = shift_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.shear_range = shear_range

        # 1. Random Rotation
        self.rotation = tf.keras.layers.RandomRotation(
            factor=rotation_range,
            fill_mode="constant",
            fill_value=0.0
        )

        # 2. Horizontal Flip
        self.flip = tf.keras.layers.RandomFlip(mode="horizontal")

        # 3. Random Zoom
        self.zoom = tf.keras.layers.RandomZoom(
            height_factor=zoom_range,
            width_factor=zoom_range,
            fill_mode="constant",
            fill_value=0.0
        )

        # 4. Translation / Shift
        self.translation = tf.keras.layers.RandomTranslation(
            height_factor=shift_range,
            width_factor=shift_range,
            fill_mode="constant",
            fill_value=0.0
        )

        # 5. Brightness adjustment
        self.brightness = tf.keras.layers.RandomBrightness(factor=brightness_range)

        # 5.5 Contrast adjustment
        self.contrast = tf.keras.layers.RandomContrast(factor=contrast_range)

        # 6. Shear
        self.shear = RandomShear(shear_factor=shear_range)

    def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
        """Executes the augmentation pipeline on inputs.

        Args:
            inputs (tf.Tensor): Preprocessed images (NHWC).
            training (bool): True to apply augmentations, False to return inputs unchanged.

        Returns:
            tf.Tensor: Augmented image tensors.
        """
        if not training:
            return inputs

        x = self.rotation(inputs, training=training)
        x = self.flip(x, training=training)
        x = self.zoom(x, training=training)
        x = self.translation(x, training=training)
        x = self.brightness(x, training=training)
        x = self.contrast(x, training=training)
        x = self.shear(x, training=training)
        return x

    def get_config(self):
        """Returns the config for layer serialization."""
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "rotation_range": self.rotation_range,
            "zoom_range": self.zoom_range,
            "shift_range": self.shift_range,
            "brightness_range": self.brightness_range,
            "contrast_range": self.contrast_range,
            "shear_range": self.shear_range,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Explicitly reconstruct from serialized config dict."""
        return cls(**config)
