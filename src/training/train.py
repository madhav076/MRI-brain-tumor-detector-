"""Training Pipeline Module.

Loads configurations, initializes datasets (tf.data), enables mixed precision,
computes class weights, and runs two-phase transfer learning (head training + fine-tuning).
Saves checkpoints, history records, parameters JSON, summary text, and metrics plots.
"""

import os
import sys

# Force UTF-8 output on Windows CMD to prevent charmap codec errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from src import config
from src.utils import set_seed, setup_logger
from src.data.dataset_loader import MRIDatasetLoader
from src.models.efficientnet_model import build_model

# Initialize logger
logger = logging.getLogger("TrainingPipeline")

def configure_mixed_precision() -> None:
    """Configures TensorFlow mixed precision policy if configured and supported."""
    if config.MIXED_PRECISION:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                # Set mixed float16 policy
                policy = tf.keras.mixed_precision.Policy("mixed_float16")
                tf.keras.mixed_precision.set_global_policy(policy)
                logger.info(
                    f"Mixed precision enabled successfully. Global policy: {tf.keras.mixed_precision.global_policy().name}"
                )
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision policy: {e}. Falling back to float32.")
        else:
            logger.info("Mixed precision requested, but no GPU devices detected. Using float32 policy.")
    else:
        logger.info("Mixed precision disabled in configs. Using standard float32 policy.")

def create_tf_dataset(
    df: pd.DataFrame,
    classes: list,
    batch_size: int,
    is_training: bool = False
) -> Optional[tf.data.Dataset]:
    """Creates a high-performance tf.data.Dataset pipeline from metadata.

    Applies decoding, resizing, pixel normalization, caching, prefetching, and
    shuffle operations.

    Args:
        df (pd.DataFrame): Metadata DataFrame.
        classes (list): List of detected class strings.
        batch_size (int): Batch size.
        is_training (bool): If True, applies shuffle.

    Returns:
        Optional[tf.data.Dataset]: Input dataset pipeline, or None if empty.
    """
    if df is None or df.empty:
        return None

    file_paths = df["file_path"].tolist()
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    labels = [class_to_idx[cls] for cls in df["class"]]

    # Create dataset of paths and numeric labels
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((path_ds, label_ds))

    # Define loading operation
    def load_and_preprocess(file_path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Read file raw bytes
        img_raw = tf.io.read_file(file_path)
        # Decode image (png/jpg/bmp)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        # Resize image
        img = tf.image.resize(img, size=config.IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        # Normalize scale to [0,1]
        img = tf.cast(img, tf.float32) / 255.0
        # Set static shape for Keras compatibility
        img.set_shape(config.IMAGE_SIZE + (3,))
        
        # One-hot encode label
        label_onehot = tf.one_hot(label, depth=config.NUM_CLASSES)
        return img, label_onehot

    # Map preprocess function
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # MED-04 FIX: Cache BEFORE batch so individual samples are cached,
    # then shuffle AFTER cache so each epoch gets a fresh random order,
    # then batch and prefetch. This avoids locked shuffle after first epoch.
    dataset = dataset.cache()

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 1000), seed=config.SEED)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def calculate_class_weights(df: pd.DataFrame, classes: list) -> Dict[int, float]:
    """Calculates class weights to handle training class imbalances.

    Args:
        df (pd.DataFrame): Training metadata DataFrame.
        classes (list): List of detected class strings.

    Returns:
        Dict[int, float]: Mapping of class integer indices to weight scales.
    """
    if df.empty:
        return {}

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    y_indices = np.array([class_to_idx[cls] for cls in df["class"]])
    unique_classes = np.unique(y_indices)
    
    # Compute balanced weights
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_indices
    )
    
    class_weights_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}
    logger.info(f"Computed class weights: {class_weights_dict}")
    return class_weights_dict

def configure_callbacks(checkpoint_path: Path) -> list:
    """Configures training callbacks: early stopping, model checkpoint, LR reduction, CSV logger.

    TensorBoard callback is intentionally excluded — it requires the tensorboard
    package to be installed separately. Use CSVLogger + manual plots instead.

    Args:
        checkpoint_path (Path): Target directory to save checkpoints.

    Returns:
        list: List of Keras Callback objects.
    """
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(checkpoint_path / "history.csv"),
            separator=",",
            append=True
        ),
    ]
    return callbacks

def save_plots(history: Dict[str, list], output_dir: Path) -> None:
    """Generates and saves performance charts from combined history.

    Args:
        history (Dict[str, list]): Combined history dictionary.
        output_dir (Path): Output directory for plots.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["loss"]) + 1)

    # 1. Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["accuracy"], "b-", label="Training Accuracy")
    plt.plot(epochs, history["val_accuracy"], "r-", label="Validation Accuracy")
    plt.title("Training & Validation Accuracy vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy.png")
    plt.close()

    # 2. Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["loss"], "b-", label="Training Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Validation Loss")
    plt.title("Training & Validation Loss vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "loss.png")
    plt.close()

    # 3. Learning Rate plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history.get("lr", [config.LEARNING_RATE] * len(epochs)), "g-", label="Learning Rate")
    plt.title("Learning Rate vs Epoch")
    plt.xlabel("Epochs")
    plt.ylabel("Learning Rate")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_rate.png")
    plt.close()
    
    logger.info(f"Saved metric graphs (accuracy.png, loss.png, learning_rate.png) to {output_dir.resolve()}")

def fine_tune_backbone(model: tf.keras.Model) -> tf.keras.Model:
    """Unfreezes the last specified layers of EfficientNetB0 backbone.

    Args:
        model (tf.keras.Model): Model instance.

    Returns:
        tf.keras.Model: Model with fine-tune layers trainable.
    """
    # Find pre-trained base model layer
    base_model = None
    for layer in model.layers:
        if layer.name.startswith("efficientnet"):
            base_model = layer
            break

    if base_model is None:
        logger.warning("EfficientNet base layer not found in model structure. Fine-tuning skipped.")
        return model

    logger.info(f"Unfreezing EfficientNetB0 backbone layers (unfreezing last {config.FINE_TUNE_LAYERS} layers)...")
    base_model.trainable = True

    # Freeze all layers except the last N
    num_layers = len(base_model.layers)
    fine_tune_at = max(0, num_layers - config.FINE_TUNE_LAYERS)
    
    # Freeze the initial layers and unfreeze fine-tune layer slice
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
        
    for layer in base_model.layers[fine_tune_at:]:
        # BatchNormalization layers must remain frozen during fine-tuning to preserve statistics
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Re-compile model with fine-tune learning rate
    metrics = [
        tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
    model.compile(
        optimizer=Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=metrics
    )
    logger.info("Successfully re-compiled model for Phase 2: Fine-Tuning.")
    return model

def run_train_pipeline() -> None:
    """Executes the entire training pipeline."""
    # Start timer
    start_time = time.time()
    
    # 1. Setup Logger & Reproduction Seed
    setup_logger(config.LOG_DIR)
    set_seed(config.SEED)
    logger.info("=" * 60)
    logger.info("           STARTING MODEL TRAINING PIPELINE           ")
    logger.info("=" * 60)

    # 2. Config Mixed Precision
    configure_mixed_precision()

    # Create check directories
    checkpoint_dir = Path(config.CHECKPOINT_DIR)
    output_dir = Path(config.OUTPUT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training").mkdir(parents=True, exist_ok=True)

    # Save Configuration parameters JSON
    config_dict = config.get_config_summary()
    with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4)
    logger.info(f"Saved configuration config settings to {checkpoint_dir / 'training_config.json'}")

    # 3. Load Datasets metadata
    logger.info(f"Scanning dataset folders under: {config.DATASET_PATH}")
    loader = MRIDatasetLoader(config.DATASET_PATH)
    metadata = loader.scan_dataset()

    # Determine if we have valid images
    has_train = "train" in metadata and not metadata["train"].empty
    has_val = "validation" in metadata and not metadata["validation"].empty

    if not (has_train and has_val):
        logger.error(
            "Dataset folder splits 'train' and 'validation' are empty or missing. "
            "Skipping training loops. Creating dummy outputs for structure verification."
        )
        # Create empty model for export checks
        model = build_model()
        
        # Save model summary text file
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        # Replace Unicode box-drawing chars (from Keras summary) with ASCII
        summary_str = "\n".join(stringlist)
        summary_str = summary_str.encode("ascii", errors="replace").decode("ascii")
        with open(checkpoint_dir / "model_summary.txt", "w", encoding="utf-8") as f:
            f.write(summary_str)
            
        logger.info(f"Saved model summary text structure to {checkpoint_dir / 'model_summary.txt'}")
        logger.warning("Pipeline terminated gracefully before training loop due to empty dataset.")
        return

    train_df = metadata["train"]
    val_df = metadata["validation"]
    logger.info(f"Training dataset size: {len(train_df)} valid images.")
    logger.info(f"Validation dataset size: {len(val_df)} valid images.")

    # 4. Generate high performance tf.data pipelines
    train_dataset = create_tf_dataset(train_df, loader.classes, config.BATCH_SIZE, is_training=True)
    val_dataset = create_tf_dataset(val_df, loader.classes, config.BATCH_SIZE, is_training=False)

    # 5. Calculate class weights
    class_weights = calculate_class_weights(train_df, loader.classes)

    # 6. Build model
    model = build_model()
    
    # Save model summary text file
    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    # Replace Unicode box-drawing chars (from Keras summary) with ASCII
    summary_str = "\n".join(stringlist)
    summary_str = summary_str.encode("ascii", errors="replace").decode("ascii")
    with open(checkpoint_dir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_str)
    logger.info(f"Model summary written to {checkpoint_dir / 'model_summary.txt'}")

    # Set callbacks
    callbacks = configure_callbacks(checkpoint_dir)

    # 7. PHASE 1: Train classification head (backbone frozen)
    logger.info(f"--- PHASE 1: Training Custom Head (Backbone Frozen) for {config.EPOCHS} epochs ---")
    
    phase1_start = time.time()
    history_phase1 = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    phase1_duration = time.time() - phase1_start
    logger.info(f"Phase 1 head training finished in {phase1_duration:.2f} seconds.")

    # Convert History object to dictionary
    hist_dict = {k: [float(val) for val in v] for k, v in history_phase1.history.items()}

    # 8. PHASE 2: Fine-Tuning (Unfreeze last layers)
    if config.FINE_TUNE_EPOCHS > 0:
        logger.info(f"--- PHASE 2: Fine-Tuning backbone (Unfreezing last {config.FINE_TUNE_LAYERS} layers) ---")
        
        # Unfreeze and compile
        model = fine_tune_backbone(model)
        
        # Total epochs offset for continuous training metrics tracking
        initial_epoch = len(hist_dict.get("loss", []))
        total_epochs = initial_epoch + config.FINE_TUNE_EPOCHS

        phase2_start = time.time()
        history_phase2 = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=total_epochs,
            initial_epoch=initial_epoch,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        phase2_duration = time.time() - phase2_start
        logger.info(f"Phase 2 fine-tuning training finished in {phase2_duration:.2f} seconds.")

        # Merge Phase 2 history into Phase 1 logs
        for k, v in history_phase2.history.items():
            if k in hist_dict:
                hist_dict[k].extend([float(val) for val in v])
            else:
                hist_dict[k] = [float(val) for val in v]

    # Save last model checkpoint
    model.save(checkpoint_dir / "last_model.keras")
    logger.info(f"Saved last model weights to {checkpoint_dir / 'last_model.keras'}")

    # MED-02 FIX: Use JSON instead of pickle for history serialization.
    # hist_dict contains only plain Python dicts/lists/floats, which are JSON-serializable.
    # Pickle files can execute arbitrary code on load, making them a security risk.
    with open(checkpoint_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(hist_dict, f, indent=2)
    logger.info(f"Saved complete training history to {checkpoint_dir / 'history.json'}")

    # Generate and save metric plots
    save_plots(hist_dict, output_dir / "training")

    # Final execution logs
    total_duration = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"TRAINING PIPELINE FINISHED SUCCESSFULLY.")
    logger.info(f"Total Pipeline Execution Time: {total_duration / 60:.2f} minutes.")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_train_pipeline()
