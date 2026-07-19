"""
fix_and_train.py  —  One-shot repair + training launcher.

Fixes:
  1. Removes non-image files from dataset/validation/
  2. Creates validation split from training data using stratified 15% sampling
  3. Verifies the complete dataset structure
  4. Runs the full training pipeline

Run from the project root:
    python fix_and_train.py
"""

import sys
import os

# -----------------------------------------------------------------------
# Force UTF-8 on stdout/stderr immediately — must happen before any import
# that might print Unicode characters.
# -----------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

os.environ["PYTHONIOENCODING"] = "utf-8"

import shutil
import random
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------
# Basic print logger (before the real logger is set up)
# -----------------------------------------------------------------------
def log(msg: str, level: str = "INFO"):
    tag = {"INFO": "[INFO]", "OK": "[ OK ]", "WARN": "[WARN]", "ERR": "[ERR ]"}.get(level, "[INFO]")
    line = f"{tag} {msg}"
    try:
        print(line)
    except UnicodeEncodeError:
        print(line.encode("ascii", errors="replace").decode("ascii"))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
VAL_RATIO = 0.15  # 15 % of training images become validation
SEED = 42


# =======================================================================
# STEP 1 — Clean validation folder and build validation split
# =======================================================================


def clean_and_build_validation(dataset_dir: Path) -> bool:
    """Remove non-image files from validation/ and build a proper split."""
    val_dir = dataset_dir / "validation"
    train_dir = dataset_dir / "train"
    val_dir.mkdir(parents=True, exist_ok=True)

    # --- 1a. Remove any non-image / non-directory items ---
    removed = []
    for item in val_dir.iterdir():
        if item.is_file() and item.suffix.lower() not in IMAGE_EXTS:
            log(f"Removing non-image file from validation/: {item.name}", "WARN")
            item.unlink()
            removed.append(item.name)
        elif item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            pass  # image file at root level — leave for now
    if removed:
        log(f"Removed {len(removed)} non-image file(s): {removed}", "OK")

    # --- 1b. Check if validation already has class dirs with images ---
    existing_val_counts = {}
    for cls in CLASSES:
        cls_dir = val_dir / cls
        if cls_dir.exists():
            imgs = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            existing_val_counts[cls] = len(imgs)

    total_existing = sum(existing_val_counts.values())
    if total_existing > 0:
        log(f"Validation already contains {total_existing} images. Skipping re-split.", "OK")
        for cls, cnt in existing_val_counts.items():
            log(f"  validation/{cls}: {cnt} images")
        return True

    # --- 1c. Build validation by stratified sampling from train ---
    log("Building validation split (15% stratified sample from train)...")
    random.seed(SEED)
    total_moved = 0

    for cls in CLASSES:
        src_dir = train_dir / cls
        dst_dir = val_dir / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            log(f"  train/{cls}/ does not exist — skipping", "WARN")
            continue

        images = [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not images:
            log(f"  train/{cls}/ is empty — skipping", "WARN")
            continue

        random.shuffle(images)
        n_val = max(1, int(len(images) * VAL_RATIO))
        val_images = images[:n_val]

        for img in val_images:
            dst = dst_dir / img.name
            if not dst.exists():
                # MOVE (not copy) to keep train set clean and avoid duplicates
                shutil.move(str(img), str(dst))

        # Count what's now in val
        val_count = len(
            [f for f in dst_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        )
        train_remaining = len(
            [f for f in src_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        )
        log(f"  {cls}: moved {n_val} to validation/ (train remaining: {train_remaining})")
        total_moved += n_val

    log(f"Validation split created — {total_moved} images total.", "OK")
    return total_moved > 0


# =======================================================================
# STEP 2 — Verify complete dataset structure
# =======================================================================


def verify_dataset(dataset_dir: Path) -> bool:
    """Print counts and return True if all splits+classes have images."""
    log("\n--- Dataset Verification ---")
    ok = True
    for split in ["train", "validation", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            log(f"  MISSING: dataset/{split}/", "ERR")
            ok = False
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                log(f"  MISSING: dataset/{split}/{cls}/", "ERR")
                ok = False
                continue
            imgs = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
            count = len(imgs)
            status = "OK" if count > 0 else "ERR"
            log(f"  dataset/{split}/{cls}: {count} images", status)
            if count == 0:
                ok = False
    return ok


# =======================================================================
# STEP 3 — Patch train.py to be fully UTF-8 safe
# =======================================================================


def patch_train_py(train_py: Path) -> None:
    """Patch src/training/train.py to write model summary with UTF-8 encoding
    and replace Unicode box-drawing chars with ASCII equivalents."""
    if not train_py.exists():
        log(f"Cannot find {train_py} — skipping patch", "WARN")
        return

    original = train_py.read_text(encoding="utf-8", errors="replace")
    patched = original

    # Fix 1: open model_summary.txt with explicit utf-8 encoding
    OLD_SUMMARY_WRITE = 'with open(checkpoint_dir / "model_summary.txt", "w") as f:'
    NEW_SUMMARY_WRITE = 'with open(checkpoint_dir / "model_summary.txt", "w", encoding="utf-8", errors="replace") as f:'
    patched = patched.replace(OLD_SUMMARY_WRITE, NEW_SUMMARY_WRITE)

    # Fix 2: sanitise the model summary string itself — replace box-drawing chars
    # Insert a sanitise step right before the summary is written
    OLD_SUMMARY_JOIN = '    summary_str = "\\n".join(stringlist)\n    with open(checkpoint_dir / "model_summary.txt"'
    NEW_SUMMARY_JOIN = (
        '    summary_str = "\\n".join(stringlist)\n'
        "    # Replace Unicode box-drawing chars with ASCII equivalents\n"
        '    summary_str = summary_str.encode("ascii", errors="replace").decode("ascii")\n'
        '    with open(checkpoint_dir / "model_summary.txt"'
    )
    patched = patched.replace(OLD_SUMMARY_JOIN, NEW_SUMMARY_JOIN)

    # Fix 3: open training_config.json with utf-8
    OLD_CONFIG_WRITE = 'with open(checkpoint_dir / "training_config.json", "w") as f:'
    NEW_CONFIG_WRITE = (
        'with open(checkpoint_dir / "training_config.json", "w", encoding="utf-8") as f:'
    )
    patched = patched.replace(OLD_CONFIG_WRITE, NEW_CONFIG_WRITE)

    # Fix 4: ensure model.save path uses forward slashes (avoids rare Windows path issues)
    # Nothing to change here — pathlib handles it fine.

    if patched != original:
        train_py.write_text(patched, encoding="utf-8")
        log(f"Patched {train_py.name} with UTF-8 safe file writes.", "OK")
    else:
        log(
            f"{train_py.name} already has UTF-8 safe file writes (or patterns not matched — check manually).",
            "WARN",
        )


def patch_utils_init(utils_init: Path) -> None:
    """Ensure setup_logger forces utf-8 on the stream handler."""
    if not utils_init.exists():
        log(f"Cannot find {utils_init} — skipping", "WARN")
        return
    text = utils_init.read_text(encoding="utf-8", errors="replace")
    # Already patched in a previous session — just verify the reconfigure call is present
    if "reconfigure" in text:
        log("src/utils/__init__.py already has UTF-8 reconfigure — no patch needed.", "OK")
    else:
        log("src/utils/__init__.py missing reconfigure — please recheck previous patches.", "WARN")


# =======================================================================
# STEP 4 — Run training pipeline
# =======================================================================


def run_training() -> bool:
    """Imports and runs the training pipeline. Returns True on success."""
    log("\n--- Starting Training Pipeline ---")
    try:
        from src.training.train import run_train_pipeline

        run_train_pipeline()
        return True
    except Exception as exc:
        import traceback

        log(f"Training failed with exception: {exc}", "ERR")
        traceback.print_exc()
        return False


# =======================================================================
# STEP 5 — Final report
# =======================================================================


def final_report(dataset_dir: Path, model_path: Path) -> None:
    log("\n" + "=" * 60)
    log("FINAL REPORT")
    log("=" * 60)
    log(f"Dataset location : {dataset_dir.resolve()}")

    for split in ["train", "validation", "test"]:
        split_dir = dataset_dir / split
        log(f"\n  {split.upper()}:")
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                n = len(
                    [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
                )
                log(f"    {cls}: {n} images")
            else:
                log(f"    {cls}: MISSING", "ERR")

    log(f"\nTraining status  : {'COMPLETED' if model_path.exists() else 'FAILED'}")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        log(f"best_model.keras : {model_path.resolve()}  ({size_mb:.1f} MB)", "OK")
    else:
        log(f"best_model.keras : NOT FOUND at {model_path.resolve()}", "ERR")


# =======================================================================
# MAIN
# =======================================================================


def main():
    log("=" * 60)
    log("Brain MRI — Dataset Repair + Training Launcher")
    log("=" * 60)

    dataset_dir = PROJECT_ROOT / "dataset"
    model_path = PROJECT_ROOT / "saved_models" / "best_model.keras"

    # ---- Step 1: Fix validation split ----
    log("\n[STEP 1] Fixing validation split...")
    ok = clean_and_build_validation(dataset_dir)
    if not ok:
        log("Could not build a valid validation split. Aborting.", "ERR")
        sys.exit(1)

    # ---- Step 2: Verify full dataset ----
    log("\n[STEP 2] Verifying dataset structure...")
    if not verify_dataset(dataset_dir):
        log("Dataset verification failed. Fix the structure and retry.", "ERR")
        sys.exit(1)
    log("Dataset structure verified.", "OK")

    # ---- Step 3: Apply UTF-8 patches ----
    log("\n[STEP 3] Applying UTF-8 safety patches...")
    patch_train_py(PROJECT_ROOT / "src" / "training" / "train.py")
    patch_utils_init(PROJECT_ROOT / "src" / "utils" / "__init__.py")

    # ---- Step 4: Train ----
    log("\n[STEP 4] Running training pipeline...")
    success = run_training()

    # ---- Step 5: Report ----
    final_report(dataset_dir, model_path)

    if not success or not model_path.exists():
        log("Training did not complete successfully. See errors above.", "ERR")
        sys.exit(1)

    log("\nAll done! Application is ready to launch.", "OK")
    log("Run:  python -m streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
