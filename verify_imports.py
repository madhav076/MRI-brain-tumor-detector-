import sys

sys.path.insert(0, ".")

print("=" * 60)
print("STATIC IMPORT VERIFICATION")
print("=" * 60)

errors = []

# Test 1: src.config
try:
    from src import config

    print(
        f"[OK] src.config loaded — IMAGE_SIZE={config.IMAGE_SIZE}, NUM_CLASSES={config.NUM_CLASSES}"
    )
except Exception as e:
    print(f"[FAIL] src.config: {e}")
    errors.append(("src.config", e))

# Test 2: src.utils
try:
    from src.utils import set_seed, setup_logger

    print("[OK] src.utils.set_seed, setup_logger")
except Exception as e:
    print(f"[FAIL] src.utils: {e}")
    errors.append(("src.utils", e))

# Test 3: src.data.augmentation (CRITICAL-01 fix)
try:
    from src.data.augmentation import MRIAugmentationPipeline, RandomShear

    print("[OK] src.data.augmentation — MRIAugmentationPipeline, RandomShear")
except Exception as e:
    print(f"[FAIL] src.data.augmentation: {e}")
    errors.append(("src.data.augmentation", e))

# Test 4: src.data.preprocessing
try:
    from src.data.preprocessing import preprocess_single_image, validate_image

    print("[OK] src.data.preprocessing")
except Exception as e:
    print(f"[FAIL] src.data.preprocessing: {e}")
    errors.append(("src.data.preprocessing", e))

# Test 5: src.data.dataset_loader
try:
    from src.data.dataset_loader import MRIDatasetLoader

    print("[OK] src.data.dataset_loader")
except Exception as e:
    print(f"[FAIL] src.data.dataset_loader: {e}")
    errors.append(("src.data.dataset_loader", e))

# Test 6: src.models.efficientnet_model
try:
    from src.models.efficientnet_model import build_model

    print("[OK] src.models.efficientnet_model")
except Exception as e:
    print(f"[FAIL] src.models.efficientnet_model: {e}")
    errors.append(("src.models.efficientnet_model", e))

# Test 7: GradCAM (CRITICAL-04 fix)
try:
    from src.evaluation.explainability.gradcam import GradCAMExplainer

    print("[OK] src.evaluation.explainability.gradcam — GradCAMExplainer")
except Exception as e:
    print(f"[FAIL] GradCAMExplainer: {e}")
    errors.append(("GradCAMExplainer", e))

# Test 8: evaluate.py (CRITICAL-05, HIGH-11 fix)
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location("evaluate", "src/evaluation/evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("[OK] src.evaluation.evaluate (all imports resolved)")
except Exception as e:
    print(f"[FAIL] src.evaluation.evaluate: {e}")
    errors.append(("src.evaluation.evaluate", e))

# Test 9: app components
try:
    from app.components.prediction_card import execute_inference, render_prediction_results

    print("[OK] app.components.prediction_card")
except Exception as e:
    print(f"[FAIL] app.components.prediction_card: {e}")
    errors.append(("prediction_card", e))

try:
    from app.components.gradcam_viewer import render_gradcam_viewer

    print("[OK] app.components.gradcam_viewer")
except Exception as e:
    print(f"[FAIL] app.components.gradcam_viewer: {e}")
    errors.append(("gradcam_viewer", e))

try:
    from app.components.report_generator import add_to_history, generate_pdf_report

    print("[OK] app.components.report_generator")
except Exception as e:
    print(f"[FAIL] app.components.report_generator: {e}")
    errors.append(("report_generator", e))

try:
    from app.components.uploader import validate_uploaded_image

    print("[OK] app.components.uploader")
except Exception as e:
    print(f"[FAIL] app.components.uploader: {e}")
    errors.append(("uploader", e))

# Test 10: src.training.train
try:
    import importlib.util

    spec = importlib.util.spec_from_file_location("train", "src/training/train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("[OK] src.training.train (all imports resolved)")
except Exception as e:
    print(f"[FAIL] src.training.train: {e}")
    errors.append(("src.training.train", e))

print()
print("=" * 60)
if errors:
    print(f"RESULT: {len(errors)} IMPORT ERROR(S) FOUND:")
    for name, err in errors:
        print(f"  - {name}: {err}")
else:
    print("RESULT: ALL IMPORTS PASSED SUCCESSFULLY")
print("=" * 60)
