"""Verification script for Module 1.

Verifies imports, config loading, dataset loader behavior on empty folder structures,
and tests preprocessing and data augmentation layers with synthetic dummy tensors.
"""

import sys
import logging
from pathlib import Path
import tensorflow as tf

# Setup path to import local packages
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.utils import setup_logger, set_seed
from src.data.dataset_loader import MRIDatasetLoader
from src.data.preprocessing import (
    validate_image,
    convert_to_rgb,
    resize_image,
    normalize_image,
    preprocess_single_image,
    preprocess_batch,
)
from src.data.augmentation import MRIAugmentationPipeline


def run_verification():
    # 1. Initialize Logger
    setup_logger(log_dir=config.LOG_DIR)
    logger = logging.getLogger("Verification")
    logger.info("Starting Module 1 verification...")

    # Set reproducibility seed
    set_seed(config.SEED)

    # 2. Check Directories existence
    logger.info("Checking folder structures...")
    directories = [
        "dataset/train",
        "dataset/validation",
        "dataset/test",
        "configs",
        "logs",
        "saved_models",
        "outputs",
        "notebooks",
    ]
    for directory in directories:
        dir_path = PROJECT_ROOT / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory check: {directory} -> Verified (Created if missing)")

    # 3. Test Config Load
    logger.info(f"Checking Configurations...")
    config_summary = config.get_config_summary()
    for k, v in config_summary.items():
        logger.info(f"  {k}: {v}")

    # 4. Initialize Dataset Loader
    logger.info("Initializing Dataset Loader...")
    loader = MRIDatasetLoader(config.DATASET_PATH)

    # Run scan on empty structure (or whatever files are there)
    metadata = loader.scan_dataset()
    logger.info("Dataset statistics computed:")
    loader.print_summary()

    # 5. Programmatic Validation of Preprocessing on Dummy Tensors
    logger.info("Running programmatic checks on preprocessing...")
    try:
        # Create a mock grayscale image slice of shape (256, 256) as uint8
        dummy_grayscale = tf.random.uniform(shape=(256, 256), minval=0, maxval=255, dtype=tf.int32)
        dummy_grayscale = tf.cast(dummy_grayscale, dtype=tf.uint8)

        # Validate
        is_valid = validate_image(dummy_grayscale)
        logger.info(f"  Image Validation: {is_valid}")

        # RGB Conversion
        rgb_tensor = convert_to_rgb(dummy_grayscale)
        logger.info(f"  Grayscale converted to RGB shape: {rgb_tensor.shape}")
        assert rgb_tensor.shape == (256, 256, 3), "RGB Conversion failed."

        # Resize
        resized_tensor = resize_image(rgb_tensor, config.IMAGE_SIZE)
        logger.info(f"  Resized image shape: {resized_tensor.shape}")
        assert resized_tensor.shape == (
            config.IMAGE_SIZE[0],
            config.IMAGE_SIZE[1],
            3,
        ), "Resize failed."

        # Normalization
        norm_tensor = normalize_image(resized_tensor, method="minmax_01")
        logger.info(
            f"  Normalized image value range: Min={tf.reduce_min(norm_tensor):.4f}, Max={tf.reduce_max(norm_tensor):.4f}"
        )

        # End-to-end preprocessing of single image
        preprocessed_single = preprocess_single_image(
            dummy_grayscale.numpy(), target_size=config.IMAGE_SIZE, normalize_method="minmax_01"
        )
        logger.info(f"  Preprocessed single image shape: {preprocessed_single.shape}")

        # Preprocess batch
        # Mock a batch of 4 grayscale images
        dummy_batch = tf.random.uniform(
            shape=(4, 256, 256, 1), minval=0, maxval=255, dtype=tf.float32
        )
        preprocessed_batch = preprocess_batch(
            dummy_batch, target_size=config.IMAGE_SIZE, normalize_method="minmax_01"
        )
        logger.info(f"  Preprocessed batch image shape: {preprocessed_batch.shape}")
        assert preprocessed_batch.shape == (
            4,
            config.IMAGE_SIZE[0],
            config.IMAGE_SIZE[1],
            3,
        ), "Batch preprocessing failed."

        logger.info("Preprocessing checks completed successfully!")

    except Exception as e:
        logger.error(f"Preprocessing check failed: {e}", exc_info=True)
        sys.exit(1)

    # 6. Programmatic Validation of Augmentation Pipeline on Dummy Tensors
    logger.info("Running programmatic checks on data augmentation...")
    try:
        augmentation_pipeline = MRIAugmentationPipeline(
            image_size=config.IMAGE_SIZE,
            rotation_range=0.15,
            zoom_range=0.1,
            shift_range=0.1,
            brightness_range=0.15,
            shear_range=0.1,
        )

        # Apply pipeline to preprocessed batch in training mode
        augmented_batch = augmentation_pipeline(preprocessed_batch, training=True)
        logger.info(f"  Augmented batch shape: {augmented_batch.shape}")
        assert (
            augmented_batch.shape == preprocessed_batch.shape
        ), "Augmentation shape changed mismatch."

        # Apply pipeline in evaluation mode (should pass images untouched)
        eval_batch = augmentation_pipeline(preprocessed_batch, training=False)
        logger.info(f"  Augmentation in eval mode checks out. Shapes: {eval_batch.shape}")

        # Test exact match in eval mode
        tf.debugging.assert_near(
            preprocessed_batch, eval_batch, message="Eval mode modified image values!"
        )
        logger.info("Augmentation checks completed successfully!")

    except Exception as e:
        logger.error(f"Augmentation check failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MODULE 1 VERIFICATION COMPLETED SUCCESSFULLY!")
    logger.info("The environment, source modules, and parameters are fully operational.")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_verification()
