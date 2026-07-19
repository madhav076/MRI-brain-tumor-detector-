"""EfficientNetB0 Transfer Learning Model Module.

Defines the CNN model architecture using the pre-trained EfficientNetB0 backbone,
integrates the custom Keras augmentation layer directly, and compiles the model.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.metrics import AUC, CategoricalAccuracy, Precision, Recall
from tensorflow.keras.optimizers import Adam

from src import config
from src.data.augmentation import MRIAugmentationPipeline

# Set up logging
logger = logging.getLogger(__name__)


def build_model(
    input_shape: Tuple[int, int, int] = config.IMAGE_SIZE + (3,),
    num_classes: int = config.NUM_CLASSES,
    learning_rate: float = config.LEARNING_RATE,
) -> models.Model:
    """Builds and compiles the EfficientNetV2B0 classification model.

    Initializes the model with the pre-trained EfficientNetV2B0 backbone frozen,
    prepends the custom Keras MRI augmentation layer, adds a custom
    classification head, and compiles it with standard optimizer, loss, and tracking metrics.

    Args:
        input_shape (Tuple[int, int, int]): Dimensions of input images (H, W, C).
        num_classes (int): Number of target classification categories.
        learning_rate (float): Optimiser starting learning rate.

    Returns:
        models.Model: Compiled Keras model instance.
    """
    logger.info(f"Building EfficientNetV2B0 model with shape {input_shape}...")
    try:
        # Initialize EfficientNetV2B0 backbone pretrained on ImageNet
        base_model = EfficientNetV2B0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )

        # Freeze backbone to preserve pretrained features during initial phase
        base_model.trainable = False
        logger.info("EfficientNetV2B0 base backbone frozen.")

        # Build model inputs
        inputs = layers.Input(shape=input_shape, name="input_image")

        # 1. Augmentation Layer (runs on GPU during training, bypassed in validation/test)
        augmentation_layer = MRIAugmentationPipeline(
            image_size=config.IMAGE_SIZE, name="mri_augmentation"
        )
        augmented = augmentation_layer(inputs)

        # 2. Base model forward pass
        features = base_model(augmented)

        # 3. Custom Classification Head
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
        x = layers.BatchNormalization(name="batch_norm")(x)
        x = layers.Dropout(0.4, name="dropout_1")(x)
        x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_256",
        )(x)
        x = layers.Dropout(0.3, name="dropout_2")(x)

        # Output layer explicitly cast to float32 (critical for mixed precision stability)
        outputs = layers.Dense(
            num_classes, activation="softmax", dtype="float32", name="predictions"
        )(x)

        # Instantiate Model
        model = models.Model(inputs=inputs, outputs=outputs, name="Brain_MRI_Tumor_Classifier")

        # Define Metrics
        metrics = [
            CategoricalAccuracy(name="accuracy"),
            Precision(name="precision"),
            Recall(name="recall"),
            AUC(name="auc"),
        ]

        # Compile Model
        # CategoricalCrossentropy expects one-hot encoded labels
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=metrics,
        )

        logger.info(
            "Successfully built and compiled EfficientNetV2B0 model with GPU augmentations."
        )
        return model

    except Exception as e:
        logger.error(f"Error during model creation/compilation: {e}", exc_info=True)
        raise e


def load_model_robustly(model_path: Union[str, Path]) -> tf.keras.Model:
    """Loads a pre-trained Keras model from path robustly.

    Tries tf.keras.models.load_model first. If that fails (e.g., due to cross-version
    Keras/TensorFlow compatibility issues with JSON serialization or zip formats),
    it dynamically rebuilds the model graph and loads weights using H5 layer name matching.

    Args:
        model_path (Union[str, Path]): File path to the trained model checkpoint.

    Returns:
        tf.keras.Model: Loaded Keras model instance.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path.resolve()}")

    logger.info(f"Attempting to load model from: {model_path.resolve()}")
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "MRIAugmentationPipeline": MRIAugmentationPipeline,
                "RandomShear": RandomShear,
            },
            compile=False,
        )
        logger.info("Successfully loaded model using standard load_model.")
        return model
    except Exception as e:
        logger.warning(
            f"Standard load_model failed ({e}). Attempting dynamic model rebuild and name-based weight resolution..."
        )

    try:
        import h5py

        # Rebuild fresh model architecture
        model = build_model(input_shape=(224, 224, 3), num_classes=4)

        with h5py.File(model_path, "r") as f:
            weight_data = {}

            def visit_h5(name, obj):
                if isinstance(obj, h5py.Dataset):
                    parts = name.split("/")
                    if len(parts) >= 3:
                        layer_name = parts[-2]
                        weight_name = parts[-1].split(":")[0]
                        if layer_name not in weight_data:
                            weight_data[layer_name] = {}
                        weight_data[layer_name][weight_name] = (
                            obj[()] if obj.shape == () else obj[:]
                        )

            f.visititems(visit_h5)

            def load_layer_weights(layer):
                if hasattr(layer, "layers"):
                    for sub_layer in layer.layers:
                        load_layer_weights(sub_layer)
                else:
                    l_name = layer.name
                    matched_name = None
                    if l_name in weight_data:
                        matched_name = l_name
                    else:
                        for h5_layer in weight_data:
                            if (
                                h5_layer == l_name
                                or l_name.endswith(h5_layer)
                                or h5_layer.endswith(l_name)
                            ):
                                matched_name = h5_layer
                                break
                    if matched_name and len(layer.weights) > 0:
                        h5_weights = weight_data[matched_name]
                        new_weights = []
                        mismatch = False
                        for w in layer.weights:
                            w_name = w.name.split("/")[-1].split(":")[0]
                            val = None
                            if w_name in h5_weights:
                                val = h5_weights[w_name]
                            else:
                                for k in h5_weights:
                                    if k in w_name or w_name in k:
                                        val = h5_weights[k]
                                        break
                            if val is not None and val.shape == w.shape:
                                new_weights.append(val)
                            else:
                                mismatch = True
                        if not mismatch and len(new_weights) == len(layer.weights):
                            layer.set_weights(new_weights)

            load_layer_weights(model)

        logger.info(
            "Successfully loaded weights directly into rebuilt architecture by layer name matching."
        )
        return model
    except Exception as ex:
        logger.error(
            f"Failed both standard loading and dynamic weight resolution: {ex}", exc_info=True
        )
        raise ex
