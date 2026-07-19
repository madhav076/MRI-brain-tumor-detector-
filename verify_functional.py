"""
Functional verification tests - ASCII-safe version for Windows CMD.
"""

import sys, os

sys.path.insert(0, ".")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# Force UTF-8 on stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import tensorflow as tf

errors = []
passed = []


def ok(msg):
    passed.append(msg)
    print("[PASS] " + msg)


def fail(msg, e):
    errors.append((msg, e))
    print("[FAIL] " + msg + ": " + str(e))


# TEST 1: Build model
try:
    from src.models.efficientnet_model import build_model

    model = build_model(input_shape=(224, 224, 3), num_classes=4)
    assert isinstance(model, tf.keras.Model)
    assert model.output_shape == (None, 4)
    ok("build_model() output_shape=" + str(model.output_shape))
except Exception as e:
    fail("build_model()", e)
    model = None

# TEST 2: get_config round-trip
try:
    from src.data.augmentation import MRIAugmentationPipeline, RandomShear

    aug = MRIAugmentationPipeline(
        rotation_range=0.15, zoom_range=0.1, shift_range=0.1, brightness_range=0.15, shear_range=0.1
    )
    cfg = aug.get_config()
    assert "rotation_range" in cfg and isinstance(cfg["rotation_range"], float)
    assert "shear_range" in cfg
    aug2 = MRIAugmentationPipeline.from_config(cfg)
    assert aug2.rotation_range == 0.15
    ok(
        "MRIAugmentationPipeline.get_config() keys="
        + str([k for k in cfg if k not in ("name", "trainable", "dtype")])
    )
except Exception as e:
    fail("MRIAugmentationPipeline.get_config()", e)

# TEST 3: RandomShear get_config
try:
    shear = RandomShear(shear_factor=0.1)
    cfg = shear.get_config()
    assert cfg["shear_factor"] == 0.1
    shear2 = RandomShear.from_config(cfg)
    assert shear2.shear_factor == 0.1
    ok("RandomShear.get_config() shear_factor=" + str(cfg["shear_factor"]))
except Exception as e:
    fail("RandomShear.get_config()", e)

# TEST 4: RandomShear forward pass
try:
    shear = RandomShear(shear_factor=0.1)
    dummy = tf.ones((2, 224, 224, 3), dtype=tf.float32)
    out = shear(dummy, training=True)
    assert out.shape == (2, 224, 224, 3)
    ok("RandomShear forward pass shape=" + str(out.shape))
except Exception as e:
    fail("RandomShear forward pass", e)

# TEST 5: MRIAugmentationPipeline forward pass
try:
    aug = MRIAugmentationPipeline()
    dummy = tf.ones((2, 224, 224, 3), dtype=tf.float32)
    out_train = aug(dummy, training=True)
    out_infer = aug(dummy, training=False)
    assert out_train.shape == (2, 224, 224, 3)
    assert np.allclose(out_infer.numpy(), dummy.numpy())
    ok("MRIAugmentationPipeline forward pass train+infer OK")
except Exception as e:
    fail("MRIAugmentationPipeline forward pass", e)

# TEST 6: GradCAM heatmap
try:
    if model is not None:
        from src.evaluation.explainability.gradcam import GradCAMExplainer

        explainer = GradCAMExplainer()
        dummy_img = tf.ones((1, 224, 224, 3), dtype=tf.float32)
        heatmap = explainer.explain(model, dummy_img, class_idx=0)
        assert len(heatmap.shape) == 2
        assert np.min(heatmap) >= 0.0 and np.max(heatmap) <= 1.0
        ok(
            "GradCAM heatmap shape="
            + str(heatmap.shape)
            + " range=["
            + str(round(float(np.min(heatmap)), 3))
            + ","
            + str(round(float(np.max(heatmap)), 3))
            + "]"
        )
    else:
        print("[SKIP] GradCAM (model not built)")
except Exception as e:
    fail("GradCAM heatmap", e)

# TEST 7: GradCAM overlay
try:
    if model is not None:
        from src.evaluation.explainability.gradcam import GradCAMExplainer

        explainer = GradCAMExplainer()
        img_np = np.ones((224, 224, 3), dtype=np.float32) * 0.5
        hmap = np.random.rand(7, 7).astype(np.float32)
        overlaid = explainer.overlay_heatmap(img_np, hmap, alpha=0.4)
        assert overlaid.shape == (224, 224, 3)
        ok("GradCAM overlay_heatmap shape=" + str(overlaid.shape))
except Exception as e:
    fail("GradCAM overlay", e)

# TEST 8: Preprocessing
try:
    from src.data.preprocessing import preprocess_single_image

    dummy_gray = np.ones((256, 256), dtype=np.uint8) * 128
    processed = preprocess_single_image(
        dummy_gray, target_size=(224, 224), normalize_method="minmax_01"
    )
    assert processed.shape == (224, 224, 3)
    ok("preprocess_single_image shape=" + str(processed.shape))
except Exception as e:
    fail("preprocess_single_image", e)

# TEST 9: execute_inference
try:
    if model is not None:
        from app.components.prediction_card import execute_inference

        dummy_gray = np.ones((256, 256), dtype=np.uint8) * 128
        probs, duration = execute_inference(dummy_gray, model)
        assert probs.shape == (1, 4)
        assert duration > 0.0
        assert abs(float(np.sum(probs)) - 1.0) < 1e-3
        ok(
            "execute_inference probs="
            + str(probs.shape)
            + " sum="
            + str(round(float(np.sum(probs)), 4))
        )
except Exception as e:
    fail("execute_inference", e)

# TEST 10: Calibration metrics
try:
    from src.evaluation.evaluate import calculate_calibration_metrics

    y_true = np.array([0, 1])
    y_prob = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    ece, mce, _, _, _ = calculate_calibration_metrics(y_true, y_prob, num_bins=10)
    assert ece == 0.0 and mce == 0.0
    ok("calculate_calibration_metrics ECE=" + str(ece) + " MCE=" + str(mce))
except Exception as e:
    fail("calculate_calibration_metrics", e)

# TEST 11: PDF report
try:
    from app.components.report_generator import generate_pdf_report

    pdf_bytes = generate_pdf_report("test.png", "meningioma", 0.887, 0.015, "1.0.0")
    assert len(pdf_bytes) > 0
    ok("generate_pdf_report bytes=" + str(len(pdf_bytes)))
except Exception as e:
    fail("generate_pdf_report", e)

# TEST 12: MRIDatasetLoader
try:
    from src.data.dataset_loader import MRIDatasetLoader
    from pathlib import Path

    loader = MRIDatasetLoader("dataset")
    assert loader.dataset_path == Path("dataset")
    assert loader.splits == ["train", "validation", "test"]
    ok("MRIDatasetLoader initialized splits=" + str(loader.splits))
except Exception as e:
    fail("MRIDatasetLoader", e)

# SUMMARY
print()
print("=" * 60)
print("RESULTS: " + str(len(passed)) + " passed, " + str(len(errors)) + " failed")
if errors:
    print("FAILURES:")
    for name, err in errors:
        print("  [FAIL] " + name + ": " + str(err))
else:
    print("ALL FUNCTIONAL TESTS PASSED")
print("=" * 60)
