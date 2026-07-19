"""Unit tests for Streamlit application utility helpers and calibration math."""

import numpy as np
from src.evaluation.evaluate import calculate_calibration_metrics
from app.components.report_generator import generate_pdf_report


def test_calibration_ece_mce():
    """Asserts Expected Calibration Error calculations match perfect calibration properties."""
    # Mocks perfect calibration
    # y_true has 3 elements: [0, 1, 2]
    # y_prob has probabilities peaking at correct indexes with exact confidence
    y_true = np.array([0, 1])
    y_prob = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    ece, mce, bin_accs, bin_confs, bin_sizes = calculate_calibration_metrics(
        y_true, y_prob, num_bins=10
    )

    assert ece == 0.0
    assert mce == 0.0
    assert bin_accs[-1] == 1.0
    assert bin_confs[-1] == 1.0


def test_pdf_report_exporter():
    """Verifies that PDF generator outputs non-empty byte lists."""
    pdf_bytes = generate_pdf_report(
        filename="test_slice.png",
        prediction="meningioma",
        confidence=0.8872,
        duration=0.015,
        model_version="1.0.0",
    )

    assert len(pdf_bytes) > 0
