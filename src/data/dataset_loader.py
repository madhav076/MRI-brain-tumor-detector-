"""Dataset loader module.

Handles automatic dataset structure detection, reorganization into the required
train/validation/test split layout, image discovery, statistics computation,
corrupt file checking, and TensorFlow tf.data.Dataset creation.

Supports the following common Brain MRI dataset layouts automatically:
  Layout A: dataset/{train,validation,test}/{class}/images  (standard - no change needed)
  Layout B: dataset/{class}/images                          (flat per-class, auto-split 80/10/10)
  Layout C: dataset/Training/{class}/images                 (Kaggle-style capitalized)
  Layout D: dataset/Testing/{class}/images                  (Kaggle-style test only)
  Layout E: arbitrary nesting - recursive image discovery   (fallback scan)
"""

import logging
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# Set up logging
logger = logging.getLogger(__name__)

# Known class names for Brain MRI datasets
KNOWN_CLASSES = {"glioma", "meningioma", "pituitary", "notumor", "no_tumor", "no tumor", "normal"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10


class MRIDatasetLoader:
    """Loads, validates, and reorganises the Brain MRI Tumor dataset.

    Automatically detects common dataset layouts and reorganises them into the
    canonical split structure (train / validation / test / {class}) before scanning.
    """

    def __init__(self, dataset_path: str):
        """Initialises the MRIDatasetLoader.

        Args:
            dataset_path (str): Root path to the dataset directory.
        """
        self.dataset_path = Path(dataset_path)
        self.splits = ["train", "validation", "test"]
        self.classes: List[str] = []
        self.metadata: Dict[str, pd.DataFrame] = {}
        self.corrupted_files: Dict[str, List[str]] = {split: [] for split in self.splits}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_structure(self) -> bool:
        """Checks that the canonical split directories exist and are non-empty.

        Returns:
            bool: True if structure is valid, False otherwise.
        """
        if not self.dataset_path.exists():
            logger.error(f"Dataset root directory does not exist: {self.dataset_path.resolve()}")
            return False

        for split in self.splits:
            split_dir = self.dataset_path / split
            if not split_dir.exists():
                logger.warning(f"Split directory missing: {split_dir}")
                return False
            # Check it actually contains class subdirectories with images
            class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
            if not class_dirs:
                logger.warning(f"Split directory '{split}' has no class subdirectories.")
                return False

        logger.info(f"Dataset structure validated: {self.dataset_path.resolve()}")
        return True

    def scan_dataset(self) -> Dict[str, pd.DataFrame]:
        """Main entry point.  Detects dataset layout, reorganises if needed, then scans.

        Returns:
            Dict[str, pd.DataFrame]: Mapping of split name -> metadata DataFrame.
        """
        if not self.dataset_path.exists():
            logger.error(f"Dataset root path does not exist: {self.dataset_path.resolve()}")
            return {}

        # Auto-detect and fix layout before scanning
        layout = self._detect_layout()
        logger.info(f"Detected dataset layout: {layout}")

        if layout != "standard":
            logger.info("Reorganising dataset into standard train/validation/test structure...")
            success = self._reorganise(layout)
            if not success:
                logger.error("Dataset reorganisation failed.  Cannot proceed with training.")
                return {}

        # Standard scan
        if not self.validate_structure():
            logger.error("Scan aborted — dataset does not follow the required structure.")
            return {}

        return self._scan_standard()

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _detect_layout(self) -> str:
        """Detects which layout the dataset follows.

        Returns:
            str: One of 'standard', 'flat', 'kaggle', 'unknown'.
        """
        # Layout A — already has train/validation/test splits with class subdirs containing images
        has_standard_splits = all(
            (self.dataset_path / s).exists() for s in ["train", "validation", "test"]
        )
        if has_standard_splits:
            # Check at least one split has class dirs with actual images
            for split in self.splits:
                split_dir = self.dataset_path / split
                class_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
                for cd in class_dirs:
                    imgs = list(cd.glob("*"))
                    imgs = [f for f in imgs if f.suffix.lower() in IMAGE_EXTENSIONS]
                    if imgs:
                        logger.info("Layout A (standard splits) detected with images present.")
                        return "standard"
            # Splits exist but are empty — fall through to detect actual images elsewhere

        # Layout C/D — Kaggle-style: Training/{class}/ and Testing/{class}/
        training_dir = self.dataset_path / "Training"
        testing_dir = self.dataset_path / "Testing"
        if training_dir.exists():
            class_dirs = [d for d in training_dir.iterdir() if d.is_dir()]
            images_found = any(
                list(cd.glob("*"))
                for cd in class_dirs
                if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in cd.iterdir() if f.is_file())
            )
            if images_found or class_dirs:
                logger.info("Layout C/D (Kaggle-style Training/Testing) detected.")
                return "kaggle"

        # Layout B — flat per-class: dataset/{class}/images
        potential_class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        flat_class_dirs = [
            d
            for d in potential_class_dirs
            if d.name.lower() not in {"train", "validation", "test", "training", "testing"}
            and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in d.rglob("*") if f.is_file())
        ]
        if flat_class_dirs:
            logger.info(
                f"Layout B (flat per-class) detected. Found class dirs: {[d.name for d in flat_class_dirs]}"
            )
            return "flat"

        # Fallback: recursive scan for any images anywhere
        all_images = list(self.dataset_path.rglob("*"))
        all_images = [f for f in all_images if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]
        if all_images:
            logger.info(
                f"Layout E (arbitrary nesting) detected. Found {len(all_images)} images recursively."
            )
            return "recursive"

        logger.warning("No images found anywhere in the dataset directory.")
        return "empty"

    # ------------------------------------------------------------------
    # Reorganisation helpers
    # ------------------------------------------------------------------

    def _reorganise(self, layout: str) -> bool:
        """Reorganises images into the canonical split structure.

        Args:
            layout (str): Detected layout identifier.

        Returns:
            bool: True if reorganisation succeeded.
        """
        if layout == "kaggle":
            return self._reorganise_kaggle()
        elif layout == "flat":
            return self._reorganise_flat()
        elif layout == "recursive":
            return self._reorganise_recursive()
        elif layout == "empty":
            logger.error(
                "No MRI images were found anywhere inside the dataset directory.\n"
                "Please download and place the Brain MRI Tumor dataset under:\n"
                f"  {self.dataset_path.resolve()}\n"
                "Expected structure:\n"
                "  dataset/train/{glioma,meningioma,pituitary,notumor}/\n"
                "  dataset/validation/{glioma,meningioma,pituitary,notumor}/\n"
                "  dataset/test/{glioma,meningioma,pituitary,notumor}/\n"
                "\nAlternatively, place the raw Kaggle dataset so that:\n"
                "  dataset/Training/{glioma,meningioma,pituitary,notumor}/\n"
                "  dataset/Testing/{glioma,meningioma,pituitary,notumor}/\n"
                "exist and this script will reorganise them automatically."
            )
            return False
        return True

    def _collect_class_images(self, source_root: Path) -> Dict[str, List[Path]]:
        """Recursively collects images grouped by class name.

        Args:
            source_root (Path): Root directory to scan.

        Returns:
            Dict[str, List[Path]]: Mapping of normalised class name -> list of image paths.
        """
        class_images: Dict[str, List[Path]] = {}

        def _normalise_class(name: str) -> str:
            """Normalise common variant spellings to canonical names."""
            n = name.lower().replace(" ", "").replace("_", "").replace("-", "")
            if n in {"notumor", "normal", "none", "healthy", "negative"}:
                return "notumor"
            if n in {"glioma", "gliomatumor"}:
                return "glioma"
            if n in {"meningioma", "meningiomatumor"}:
                return "meningioma"
            if n in {"pituitary", "pituitarytumor"}:
                return "pituitary"
            return name.lower()  # keep unknown names as-is

        for item in source_root.rglob("*"):
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
                # Use the immediate parent folder name as class label
                cls = _normalise_class(item.parent.name)
                class_images.setdefault(cls, []).append(item)

        return class_images

    def _split_and_copy(self, class_images: Dict[str, List[Path]]) -> bool:
        """Splits images 80/10/10 into train/validation/test and copies them.

        Args:
            class_images (Dict[str, List[Path]]): Class -> image path list.

        Returns:
            bool: True if at least one split was populated.
        """
        random.seed(42)
        total_copied = 0

        for cls, images in class_images.items():
            random.shuffle(images)
            n = len(images)
            n_train = int(n * TRAIN_RATIO)
            n_val = int(n * VAL_RATIO)
            # Remaining images go to test
            splits = {
                "train": images[:n_train],
                "validation": images[n_train : n_train + n_val],
                "test": images[n_train + n_val :],
            }

            for split_name, split_images in splits.items():
                dest_dir = self.dataset_path / split_name / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    dest = dest_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)
                        total_copied += 1

            logger.info(
                f"Class '{cls}': {len(images)} images -> "
                f"train={len(splits['train'])}, "
                f"val={len(splits['validation'])}, "
                f"test={len(splits['test'])}"
            )

        logger.info(f"Total images copied into split structure: {total_copied}")
        return total_copied > 0

    def _reorganise_kaggle(self) -> bool:
        """Handles Kaggle-style Training/Testing layout."""
        training_dir = self.dataset_path / "Training"
        testing_dir = self.dataset_path / "Testing"

        # Collect from Training (large set) for train/val splits
        train_class_images = (
            self._collect_class_images(training_dir) if training_dir.exists() else {}
        )
        test_class_images = self._collect_class_images(testing_dir) if testing_dir.exists() else {}

        random.seed(42)
        total_copied = 0
        all_classes = set(train_class_images.keys()) | set(test_class_images.keys())

        for cls in all_classes:
            train_imgs = train_class_images.get(cls, [])
            test_imgs = test_class_images.get(cls, [])

            random.shuffle(train_imgs)
            n_train = len(train_imgs)
            n_val = max(1, int(n_train * VAL_RATIO / TRAIN_RATIO))  # carve val from train
            n_actual_train = n_train - n_val

            splits = {
                "train": train_imgs[:n_actual_train],
                "validation": train_imgs[n_actual_train:],
                "test": (
                    test_imgs
                    if test_imgs
                    else train_imgs[n_actual_train : n_actual_train + max(1, n_val // 2)]
                ),
            }

            for split_name, split_images in splits.items():
                dest_dir = self.dataset_path / split_name / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img_path in split_images:
                    dest = dest_dir / img_path.name
                    if not dest.exists():
                        shutil.copy2(img_path, dest)
                        total_copied += 1

            logger.info(
                f"[Kaggle] Class '{cls}': "
                f"train={len(splits['train'])}, "
                f"val={len(splits['validation'])}, "
                f"test={len(splits['test'])}"
            )

        logger.info(f"Kaggle reorganisation complete. Total files copied: {total_copied}")
        return total_copied > 0

    def _reorganise_flat(self) -> bool:
        """Handles flat per-class layout: dataset/{class}/images."""
        class_images = {}
        excluded = {"train", "validation", "test", "training", "testing"}
        for d in self.dataset_path.iterdir():
            if d.is_dir() and d.name.lower() not in excluded:
                imgs = [
                    f for f in d.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
                ]
                if imgs:
                    class_images[d.name.lower()] = imgs

        return self._split_and_copy(class_images)

    def _reorganise_recursive(self) -> bool:
        """Fallback: recursively collect all images, infer class from parent folder."""
        class_images = self._collect_class_images(self.dataset_path)
        # Remove split names that were accidentally picked up
        for split in ["train", "validation", "test", "training", "testing"]:
            class_images.pop(split, None)
        return self._split_and_copy(class_images)

    # ------------------------------------------------------------------
    # Standard scan (after structure is guaranteed)
    # ------------------------------------------------------------------

    def _scan_standard(self) -> Dict[str, pd.DataFrame]:
        """Scans the canonical train/validation/test split folders.

        Returns:
            Dict[str, pd.DataFrame]: Split name -> metadata DataFrame.
        """
        detected_classes: set = set()

        for split in self.splits:
            split_dir = self.dataset_path / split
            if split_dir.exists():
                for subfolder in split_dir.iterdir():
                    if subfolder.is_dir():
                        detected_classes.add(subfolder.name)

        self.classes = sorted(list(detected_classes))
        if not self.classes:
            logger.error(
                "No class subdirectories discovered in split folders after reorganisation."
            )
            return {}

        logger.info(f"Detected classes: {self.classes}")

        for split in self.splits:
            split_dir = self.dataset_path / split
            records: List[Dict[str, Any]] = []

            if not split_dir.exists():
                self.metadata[split] = pd.DataFrame()
                continue

            for class_name in self.classes:
                class_dir = split_dir / class_name
                if not class_dir.exists():
                    logger.warning(f"Class folder '{class_name}' not found in split '{split}'.")
                    continue

                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                        image_info = self._validate_and_extract_metadata(file_path, class_name)
                        if image_info:
                            records.append(image_info)
                        else:
                            self.corrupted_files[split].append(str(file_path.resolve()))

            df = pd.DataFrame(records)
            self.metadata[split] = df
            logger.info(
                f"Split '{split}': {len(df)} valid images. "
                f"Skipped {len(self.corrupted_files[split])} corrupted files."
            )

        return self.metadata

    # ------------------------------------------------------------------
    # Image validation helper
    # ------------------------------------------------------------------

    def _validate_and_extract_metadata(
        self, file_path: Path, class_name: str
    ) -> Optional[Dict[str, Any]]:
        """Validates an image file and extracts properties.

        Args:
            file_path (Path): Path to the image file.
            class_name (str): The corresponding class name.

        Returns:
            Optional[Dict[str, Any]]: Image properties dict or None if invalid.
        """
        try:
            if file_path.stat().st_size == 0:
                logger.warning(f"File {file_path.name} is empty (0 bytes).")
                return None

            with Image.open(file_path) as img:
                img.verify()

            img_arr = cv2.imread(str(file_path.resolve()), cv2.IMREAD_UNCHANGED)
            if img_arr is None:
                logger.warning(f"OpenCV failed to read: {file_path.name}")
                return None

            height, width = img_arr.shape[:2]
            channels = img_arr.shape[2] if len(img_arr.shape) == 3 else 1
            img_normalized = img_arr.astype(np.float32) / 255.0
            mean_intensity = float(np.mean(img_normalized))
            std_intensity = float(np.std(img_normalized))

            return {
                "file_path": str(file_path.resolve()),
                "filename": file_path.name,
                "class": class_name,
                "width": width,
                "height": height,
                "channels": channels,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
            }

        except Exception as e:
            logger.warning(f"Error reading/validating {file_path.name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Statistics & reporting
    # ------------------------------------------------------------------

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Calculates global and split-specific dataset statistics.

        Returns:
            Dict[str, Any]: Nested dict containing summaries per split.
        """
        stats: Dict[str, Any] = {}

        for split in self.splits:
            df = self.metadata.get(split)
            if df is None or df.empty:
                stats[split] = {"total_images": 0, "class_counts": {}}
                continue

            total_images = len(df)
            class_counts = df["class"].value_counts().to_dict()
            class_imbalance = {
                cls: {
                    "count": count,
                    "percentage": float(count / total_images) * 100,
                }
                for cls, count in class_counts.items()
            }

            min_w, max_w = int(df["width"].min()), int(df["width"].max())
            min_h, max_h = int(df["height"].min()), int(df["height"].max())
            mean_w, mean_h = float(df["width"].mean()), float(df["height"].mean())
            global_mean = float(df["mean_intensity"].mean())
            global_std = float(df["std_intensity"].mean())

            stats[split] = {
                "total_images": total_images,
                "class_counts": class_counts,
                "class_imbalance": class_imbalance,
                "resolution": {
                    "min_width": min_w,
                    "max_width": max_w,
                    "mean_width": mean_w,
                    "min_height": min_h,
                    "max_height": max_h,
                    "mean_height": mean_h,
                },
                "pixels": {"mean_intensity": global_mean, "std_intensity": global_std},
                "corrupted_count": len(self.corrupted_files.get(split, [])),
            }

        return stats

    def print_summary(self) -> None:
        """Prints a human-readable summary of dataset characteristics."""
        stats = self.get_dataset_statistics()
        print("=" * 60)
        print("       BRAIN MRI DATASET STATISTICAL SUMMARY       ")
        print("=" * 60)
        for split, split_stats in stats.items():
            print(f"\nSPLIT: {split.upper()}")
            print("-" * 30)
            print(f"Total Valid Images: {split_stats['total_images']}")
            print(f"Corrupted Files:    {split_stats.get('corrupted_count', 0)}")
            if split_stats["total_images"] > 0:
                print("\nClass Distribution:")
                for cls, details in split_stats["class_imbalance"].items():
                    print(f"  - {cls}: {details['count']} ({details['percentage']:.2f}%)")
            else:
                print("  No valid images found.")
            print("-" * 60)
