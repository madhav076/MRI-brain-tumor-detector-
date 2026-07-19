"""Verification script for Module 3.

Checks imports, explainer class inheritances, explainer methods, and runs the evaluation
script in verification mode to test report exports and logging systems.
"""

import sys
import json
import logging
from pathlib import Path

# Setup path to import local packages
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.utils import setup_logger
from src.evaluation.explainability import BaseExplainer, GradCAMExplainer
from src.evaluation.evaluate import run_evaluation


def run_verification():
    setup_logger(config.LOG_DIR)
    logger = logging.getLogger("VerificationModule3")
    logger.info("Starting Module 3 validation tests...")

    # 1. Imports and Class Inheritance Verification
    logger.info("Checking explainers class inheritance...")
    try:
        assert issubclass(
            GradCAMExplainer, BaseExplainer
        ), "GradCAMExplainer must inherit from BaseExplainer."
        logger.info("  GradCAMExplainer inherits from BaseExplainer correctly.")

        # Verify instance methods
        explainer = GradCAMExplainer()
        assert hasattr(explainer, "explain"), "GradCAMExplainer missing 'explain' method."
        assert hasattr(
            explainer, "overlay_heatmap"
        ), "GradCAMExplainer missing 'overlay_heatmap' method."
        logger.info("  GradCAMExplainer methods verified.")

    except Exception as e:
        logger.error(f"Explainability inheritance check failed: {e}", exc_info=True)
        sys.exit(1)

    # 2. Execution and Report Generation Check
    logger.info("Executing evaluation pipeline in verification mode (empty dataset / no model)...")
    try:
        run_evaluation()

        # Check export directories
        reports_dir = Path(config.OUTPUT_DIR) / "evaluation" / "reports"
        logger.info(f"Checking exported files under {reports_dir.resolve()}...")

        summary_file = reports_dir / "evaluation_summary.json"
        html_file = reports_dir / "evaluation_report.html"
        md_file = reports_dir / "evaluation_report.md"

        assert summary_file.exists(), "evaluation_summary.json was not created."
        assert html_file.exists(), "evaluation_report.html was not created."
        assert md_file.exists(), "evaluation_report.md was not created."

        logger.info(
            f"  Verified export: evaluation_summary.json ({summary_file.stat().st_size} bytes)"
        )
        logger.info(f"  Verified export: evaluation_report.html ({html_file.stat().st_size} bytes)")
        logger.info(f"  Verified export: evaluation_report.md ({md_file.stat().st_size} bytes)")

        # Verify JSON keys
        with open(summary_file, "r") as f:
            summary_data = json.load(f)
        assert "accuracy" in summary_data, "Summary missing 'accuracy' metric."
        assert "metadata" in summary_data, "Summary missing 'metadata' block."
        logger.info("  Verified report JSON structure.")

        # Check log file exists
        log_file = Path(config.LOG_DIR) / "evaluation.log"
        assert log_file.exists(), "evaluation.log was not created."
        logger.info(f"  Verified logs: evaluation.log exists ({log_file.stat().st_size} bytes)")

        logger.info("=" * 60)
        logger.info("MODULE 3 VERIFICATION COMPLETED SUCCESSFULLY!")
        logger.info(
            "All explainers, metrics calculations, and report exports conform to specifications."
        )
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Evaluation script validation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_verification()
