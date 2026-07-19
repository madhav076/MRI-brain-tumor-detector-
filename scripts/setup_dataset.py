"""Dataset Setup Helper.

Scans the project for MRI images, detects dataset layout, reorganises if needed,
and prints clear download instructions if no images are found.

Usage:
    python scripts/setup_dataset.py
"""

import sys
import os
from pathlib import Path

# Force UTF-8 stdout
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def count_images(root: Path) -> int:
    return sum(1 for f in root.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)


def scan_everywhere() -> dict:
    """Scan the entire project for image files and return counts by directory."""
    results = {}
    dataset_dir = PROJECT_ROOT / "dataset"
    if dataset_dir.exists():
        results["dataset/"] = count_images(dataset_dir)
    # Check common places users might have put the dataset
    for candidate in [
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "images",
        PROJECT_ROOT / "Training",
        PROJECT_ROOT / "Testing",
        PROJECT_ROOT.parent / "brain-tumor-mri-dataset",
        PROJECT_ROOT.parent / "Brain_Tumor_MRI",
        PROJECT_ROOT.parent / "MRI_dataset",
    ]:
        if candidate.exists():
            count = count_images(candidate)
            if count > 0:
                results[str(candidate)] = count
    return results


def try_kaggle_download() -> bool:
    """Attempt to download from Kaggle API if credentials exist."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return False
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[INFO] kaggle package not installed. Run: pip install kaggle")
        return False

    dataset_dir = PROJECT_ROOT / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    print("[INFO] Kaggle credentials found. Attempting to download Brain MRI dataset...")
    print("[INFO] Downloading: masoudnickparvar/brain-tumor-mri-dataset ...")
    try:
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", "masoudnickparvar/brain-tumor-mri-dataset",
                "--unzip", "-p", str(dataset_dir),
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("[SUCCESS] Dataset downloaded successfully.")
            return True
        else:
            print(f"[WARN] Kaggle download returned exit code {result.returncode}.")
            print(result.stderr[:500] if result.stderr else "")
            return False
    except Exception as e:
        print(f"[WARN] Kaggle download failed: {e}")
        return False


def print_instructions():
    dataset_dir = PROJECT_ROOT / "dataset"
    print()
    print("=" * 65)
    print("  BRAIN MRI DATASET NOT FOUND")
    print("=" * 65)
    print()
    print("No MRI images were found anywhere in the project.")
    print()
    print("OPTION 1 — Kaggle (recommended, free account required)")
    print("-" * 65)
    print("  1. Register at https://www.kaggle.com and go to Account > API")
    print("  2. Click 'Create New API Token' — downloads kaggle.json")
    print("  3. Place kaggle.json in: C:\\Users\\<you>\\.kaggle\\kaggle.json")
    print("  4. Run:  pip install kaggle")
    print("  5. Run:  python scripts/setup_dataset.py   (will auto-download)")
    print()
    print("OPTION 2 — Manual Download")
    print("-" * 65)
    print("  1. Go to: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset")
    print("  2. Click 'Download' (ZIP file ~180 MB)")
    print("  3. Extract the ZIP.  You should see folders named:")
    print("       Training/  and  Testing/")
    print("     (or glioma/ meningioma/ pituitary/ notumor/)")
    print("  4. Copy or move these folders into:")
    print(f"       {dataset_dir.resolve()}")
    print()
    print("  Final layout after extraction (Kaggle default):")
    print(f"    {dataset_dir}/Training/glioma/")
    print(f"    {dataset_dir}/Training/meningioma/")
    print(f"    {dataset_dir}/Training/pituitary/")
    print(f"    {dataset_dir}/Training/notumor/")
    print(f"    {dataset_dir}/Testing/glioma/")
    print(f"    {dataset_dir}/Testing/.../")
    print()
    print("  The training pipeline will automatically reorganise this")
    print("  into train / validation / test splits.")
    print()
    print("OPTION 3 — Alternative Kaggle dataset (also supported)")
    print("-" * 65)
    print("  https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri")
    print("  Same steps.  Supports Training/ and Testing/ folders.")
    print()
    print("=" * 65)
    print("  After placing the dataset, run:")
    print("    python scripts/run_training.py")
    print("=" * 65)
    print()


def main():
    print("=" * 65)
    print("  Brain MRI Dataset Setup")
    print("=" * 65)

    # Check for images everywhere
    found = scan_everywhere()

    if not any(v > 0 for v in found.values()):
        print("[INFO] No images found in project directory.")

        # Try Kaggle auto-download
        downloaded = try_kaggle_download()

        if not downloaded:
            print_instructions()
            sys.exit(1)

        # Re-check after download
        found = scan_everywhere()

    # Report what was found
    print()
    for location, count in found.items():
        if count > 0:
            print(f"  Found {count:,} images in: {location}")

    # Now verify/run the reorganisation via MRIDatasetLoader
    from src.data.dataset_loader import MRIDatasetLoader
    from src import config

    print()
    print("[INFO] Running dataset layout detection and reorganisation...")
    loader = MRIDatasetLoader(config.DATASET_PATH)
    metadata = loader.scan_dataset()

    if not metadata:
        print("[ERROR] Dataset setup failed. Please check the errors above.")
        sys.exit(1)

    loader.print_summary()

    total = sum(len(df) for df in metadata.values() if df is not None and not df.empty)
    print(f"\n[SUCCESS] Dataset ready. Total valid images: {total:,}")
    print("  Run training with:  python scripts/run_training.py")


if __name__ == "__main__":
    main()
