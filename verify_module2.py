"""Verification script for Module 2.

Checks that the model builds, the Keras layer properties match the design specifications,
and runs the training script in verification mode to test the config exports.
"""

import json
import logging
import os
import sys
from pathlib import Path

import tensorflow as tf

# Setup path to import local packages
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.models.efficientnet_model import build_model
from src.utils import set_seed, setup_logger


def run_verification():
    setup_logger(config.LOG_DIR)
    logger = logging.getLogger("VerificationModule2")
    logger.info("Starting Module 2 validation tests...")

    set_seed(config.SEED)

    # 1. Model Compilation Check
    logger.info("Testing model compilation...")
    try:
        model = build_model()
        logger.info("Model compiled successfully!")

        # 2. Check structure layers
        logger.info("Validating model layer names and properties...")
        layer_names = [layer.name for layer in model.layers]
        logger.info(f"Model layers: {layer_names}")

        # Assert mandatory components
        assert "mri_augmentation" in layer_names, "Augmentation layer missing from model."
        assert any(
            l.startswith("efficientnet") for l in layer_names
        ), "EfficientNet backbone missing."
        assert "global_avg_pool" in layer_names, "GlobalAveragePooling2D head missing."
        assert "batch_norm" in layer_names, "BatchNormalization head missing."
        assert "dropout_1" in layer_names, "First Dropout layer missing."
        assert "dense_256" in layer_names, "Dense hidden layer missing."
        assert "dropout_2" in layer_names, "Second Dropout layer missing."
        assert (
            "predictions" in layer_names
        ), "Final classification output predictions layer missing."

        # Verify output activation and dtype policy (mixed precision check)
        prediction_layer = model.get_layer("predictions")
        logger.info(f"Predictions output dtype policy: {prediction_layer.dtype_policy.name}")
        assert (
            prediction_layer.dtype_policy.name == "float32"
        ), "Final Dense layer must use float32 policy for numerical stability."

        logger.info("Model architecture validated successfully!")

    except Exception as e:
        logger.error(f"Model validation check failed: {e}", exc_info=True)
        sys.exit(1)

    # 3. Execution Check (Runs train.py to verify configurations and statistics initialization)
    logger.info("Executing training script in verification mode (empty dataset)...")
    try:
        # Import train runner directly
        from src.training.train import run_train_pipeline

        run_train_pipeline()

        # Check export files exist
        checkpoint_dir = Path(config.CHECKPOINT_DIR)
        logger.info(f"Checking exported files under {checkpoint_dir.resolve()}...")

        summary_file = checkpoint_dir / "model_summary.txt"
        config_file = checkpoint_dir / "training_config.json"

        assert summary_file.exists(), "model_summary.txt was not created."
        assert config_file.exists(), "training_config.json was not created."

        logger.info(f"  Verified export: model_summary.txt ({summary_file.stat().st_size} bytes)")

        # Load and verify JSON config
        with open(config_file, "r") as f:
            config_data = json.load(f)
        logger.info(f"  Verified export: training_config.json containing {len(config_data)} keys.")

        logger.info("=" * 60)
        logger.info("MODULE 2 VERIFICATION COMPLETED SUCCESSFULLY!")
        logger.info("All parameters, architectures, and training outputs conform to requirements.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training script validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
