"""Configuration management module.

Loads environment variables and YAML configurations for the Brain MRI Tumor
Classification project. Provides safe defaults if config file is not found.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

# Setup logging
logger = logging.getLogger(__name__)

# Base Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Loads settings from a YAML configuration file.

    Args:
        config_path (Path): Path to the config file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using fallback default settings.")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            if not isinstance(config, dict):
                logger.error(f"Invalid YAML structure at {config_path}. Must be a dictionary.")
                return {}
            logger.info(f"Successfully loaded configurations from {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error reading config file: {e}")
        return {}


# Load YAML configs
config_dict = load_yaml_config(CONFIG_PATH)

# Extract config categories with fallback defaults
project_cfg = config_dict.get("project", {})
data_cfg = config_dict.get("data", {})
training_cfg = config_dict.get("training", {})

# Global configurations with strict fallbacks
PROJECT_NAME: str = project_cfg.get("name", "Brain-MRI-Tumor-Classification")
VERSION: str = project_cfg.get("version", "1.0.0")
SEED: int = project_cfg.get("seed", 42)
DEVICE: str = project_cfg.get("device", "GPU")
NUM_WORKERS: int = project_cfg.get("num_workers", 4)

DATASET_PATH: str = str(PROJECT_ROOT / data_cfg.get("dataset_path", "dataset"))
IMAGE_SIZE: Tuple[int, int] = tuple(data_cfg.get("image_size", [224, 224]))  # type: ignore
BATCH_SIZE: int = data_cfg.get("batch_size", 32)
NUM_CLASSES: int = data_cfg.get("num_classes", 4)

MODEL_NAME: str = training_cfg.get("model_name", "EfficientNetB0")
LEARNING_RATE: float = training_cfg.get("learning_rate", 0.0001)
EPOCHS: int = training_cfg.get("epochs", 10)
FINE_TUNE_EPOCHS: int = training_cfg.get("fine_tune_epochs", 15)
FINE_TUNE_LEARNING_RATE: float = training_cfg.get("fine_tune_learning_rate", 0.00001)
FINE_TUNE_LAYERS: int = training_cfg.get("fine_tune_layers", 25)
MIXED_PRECISION: bool = training_cfg.get("mixed_precision", True)

CHECKPOINT_DIR: str = str(PROJECT_ROOT / training_cfg.get("checkpoint_dir", "saved_models"))
OUTPUT_DIR: str = str(PROJECT_ROOT / training_cfg.get("output_dir", "outputs"))
LOG_DIR: str = str(PROJECT_ROOT / training_cfg.get("log_dir", "logs"))

for required_dir in [DATASET_PATH, CHECKPOINT_DIR, OUTPUT_DIR, LOG_DIR]:
    Path(required_dir).mkdir(parents=True, exist_ok=True)


def find_model_path() -> Path:
    """Locates the pre-trained Keras model in the project directory by prioritizing
    specific names: best_model.keras, best_model.h5, model.keras, model.h5,
    final_model.keras, checkpoint.keras.

    Scans the saved_models/ folder first, then falls back to parent dirs.

    Returns:
        Path: Detected model filepath or standard default path.
    """
    saved_models_dir = PROJECT_ROOT / "saved_models"
    target_names = [
        "best_model.keras",
        "best_model.h5",
        "model.keras",
        "model.h5",
        "final_model.keras",
        "checkpoint.keras",
    ]

    # 1. Scan saved_models directory first for target names in order
    if saved_models_dir.exists():
        for name in target_names:
            model_file = saved_models_dir / name
            if model_file.exists():
                return model_file

        # Fallback to any .keras or .h5 file in saved_models
        for ext in ["*.keras", "*.h5"]:
            models = list(saved_models_dir.glob(ext))
            if models:
                return models[0]

    # 2. General recursive scan in PROJECT_ROOT (excluding virtual environments and dataset folders)
    exclude_dirs = {
        ".venv",
        "venv",
        "env",
        "dataset",
        ".git",
        "logs",
        "outputs",
        "tests",
        "docs",
        "app",
        "notebooks",
    }
    try:
        for p in PROJECT_ROOT.iterdir():
            if p.is_dir() and p.name not in exclude_dirs:
                # Check for target names first
                for name in target_names:
                    model_file = p / name
                    if model_file.exists():
                        return model_file
                # Then check extensions
                for ext in ["*.keras", "*.h5"]:
                    models = list(p.glob(ext))
                    if models:
                        return models[0]
    except Exception:
        pass

    # Default fallback path
    return saved_models_dir / "best_model.keras"


MODEL_PATH: Path = find_model_path()


def get_config_summary() -> Dict[str, Any]:
    """Returns a dictionary summary of active configuration variables.

    Returns:
        Dict[str, Any]: Active parameters summary.
    """
    return {
        "PROJECT_NAME": PROJECT_NAME,
        "VERSION": VERSION,
        "SEED": SEED,
        "DEVICE": DEVICE,
        "NUM_WORKERS": NUM_WORKERS,
        "DATASET_PATH": DATASET_PATH,
        "IMAGE_SIZE": IMAGE_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_CLASSES": NUM_CLASSES,
        "MODEL_NAME": MODEL_NAME,
        "LEARNING_RATE": LEARNING_RATE,
        "EPOCHS": EPOCHS,
        "FINE_TUNE_EPOCHS": FINE_TUNE_EPOCHS,
        "FINE_TUNE_LEARNING_RATE": FINE_TUNE_LEARNING_RATE,
        "FINE_TUNE_LAYERS": FINE_TUNE_LAYERS,
        "MIXED_PRECISION": MIXED_PRECISION,
        "CHECKPOINT_DIR": CHECKPOINT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "LOG_DIR": LOG_DIR,
        "MODEL_PATH": str(MODEL_PATH),
    }
