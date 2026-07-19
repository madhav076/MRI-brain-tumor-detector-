"""Streamlit App Report Exporter and History Component.

Manages prediction history, exports CSV/JSON summaries, and generates
downloadable PDF clinical logs.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import streamlit as st

# Setup logging
logger = logging.getLogger(__name__)

# Check FPDF availability for PDF generation
try:
    from fpdf import FPDF, XPos, YPos

    has_fpdf = True
except ImportError:
    has_fpdf = False
    logger.warning("fpdf2 package not discovered. PDF exports will fallback to text summaries.")


def init_session_history() -> None:
    """Initializes the session history log in Streamlit state."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def add_to_history(filename: str, prediction: str, confidence: float, duration: float) -> None:
    """Appends an inference prediction record to session history.

    Args:
        filename (str): Name of the scan file.
        prediction (str): Predicted diagnosis label.
        confidence (float): Confidence score value.
        duration (float): Prediction execution time.
    """
    init_session_history()

    record = {
        "Filename": filename,
        "Prediction": prediction.upper(),
        "Confidence": f"{confidence:.2%}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Duration (s)": f"{duration:.3f}",
    }

    st.session_state.prediction_history.append(record)
    logger.info(f"Added prediction record for '{filename}' to session history.")


def render_history_section() -> None:
    """Renders the prediction history data table and provides export/clear buttons."""
    init_session_history()

    st.subheader("Prediction History Logs")

    if not st.session_state.prediction_history:
        st.info("No prediction history logs recorded yet.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.prediction_history)

    # Render table
    st.dataframe(df, use_container_width=True)

    # Actions row
    col1, col2 = st.columns(2)

    with col1:
        # Export as CSV button
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History Logs (CSV)",
            data=csv_data,
            file_name="mri_prediction_history.csv",
            mime="text/csv",
        )

    with col2:
        # Clear history button
        if st.button("🗑️ Clear History Logs", key="clear_history_btn"):
            st.session_state.prediction_history = []
            logger.info("Session history logs cleared by the user.")
            st.rerun()  # CRITICAL-02: st.experimental_rerun() removed in Streamlit ≥1.27


def generate_pdf_report(
    filename: str, prediction: str, confidence: float, duration: float, model_version: str
) -> bytes:
    """Generates PDF report content as binary bytes.

    Args:
        filename (str): Target filename.
        prediction (str): Predicted class.
        confidence (float): Confidence rate.
        duration (float): Run time.
        model_version (str): Model version tag.

    Returns:
        bytes: PDF binary content.
    """
    if not has_fpdf:
        # Fallback text summary
        text_report = (
            f"BRAIN MRI CLASSIFICATION REPORT\n"
            f"===============================\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Scan File: {filename}\n"
            f"Predicted Diagnosis: {prediction.upper()}\n"
            f"Confidence: {confidence:.2%}\n"
            f"Inference Time: {duration:.3f} seconds\n"
            f"Model Version: {model_version}\n\n"
            f"MEDICAL DISCLAIMER:\n"
            f"This is an educational AI demonstration and NOT a medical diagnostic tool."
        )
        return text_report.encode("utf-8")

    try:
        pdf = FPDF()
        pdf.add_page()

        # Header banner
        pdf.set_fill_color(44, 62, 80)  # Dark Blue
        pdf.rect(0, 0, 210, 40, "F")

        # Title
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 18)
        pdf.text(15, 25, "BRAIN MRI CLASSIFICATION REPORT")

        # Reset text color
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 12)

        # Report details
        pdf.ln(45)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(190, 10, "1. Diagnostic Results", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 12)
        pdf.line(10, 52, 200, 52)

        pdf.ln(5)
        pdf.cell(50, 10, "Target Scan File:", border=0)
        pdf.cell(140, 10, filename, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(50, 10, "Predicted Diagnosis:", border=0)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(140, 10, prediction.upper(), border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 12)

        pdf.cell(50, 10, "Confidence Level:", border=0)
        pdf.cell(140, 10, f"{confidence:.2%}", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(50, 10, "Inference Duration:", border=0)
        pdf.cell(140, 10, f"{duration:.3f} seconds", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        pdf.cell(50, 10, "Timestamp Logged:", border=0)
        pdf.cell(
            140,
            10,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        # Model details
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(190, 10, "2. System Specifications", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 12)
        pdf.line(10, 117, 200, 117)

        pdf.ln(5)
        pdf.cell(50, 10, "Classification Model:", border=0)
        pdf.cell(
            140,
            10,
            "EfficientNetB0 Transfer Learning",
            border=0,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        pdf.cell(50, 10, "Model Version Tag:", border=0)
        pdf.cell(140, 10, model_version, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        # Disclaimer block
        pdf.ln(15)
        pdf.set_fill_color(248, 215, 218)  # Pink/Red alert box
        pdf.rect(10, 145, 190, 35, "F")

        pdf.set_text_color(114, 28, 36)  # Red text
        pdf.set_xy(12, 147)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(186, 6, "MEDICAL DISCLAIMER NOTICE:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_x(12)
        pdf.multi_cell(
            186,
            5,
            "This application is an educational AI demonstration project showing deep learning capabilities.\n"
            "It is NOT a medical diagnosis tool, it does not possess clinical certifications, and must NOT\n"
            "be used to replace guidance or medical assessments from qualified medical professionals.",
        )

        return bytes(pdf.output())
    except Exception as e:
        logger.error(f"Failed to generate PDF document layout: {e}", exc_info=True)
        return f"PDF generation error: {e}".encode("utf-8")


def render_exporters_section(
    filename: str, prediction: str, confidence: float, duration: float, model_version: str
) -> None:
    """Renders download buttons to export current predictions in CSV, JSON, and PDF formats.

    Args:
        filename (str): Image filename.
        prediction (str): Predicted class.
        confidence (float): Confidence score.
        duration (float): Prediction time.
        model_version (str): System version.
    """
    st.subheader("Export Results Report")

    col1, col2, col3 = st.columns(3)

    # 1. JSON Report Export
    report_dict = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "prediction": prediction.upper(),
        "confidence": confidence,
        "duration_seconds": duration,
        "model_version": model_version,
    }
    json_data = json.dumps(report_dict, indent=4)
    with col1:
        st.download_button(
            label="📥 Export Report (JSON)",
            data=json_data,
            file_name=f"report_{filename.split('.')[0]}.json",
            mime="application/json",
        )

    # 2. CSV Report Export
    csv_df = pd.DataFrame(
        [
            {
                "Timestamp": report_dict["timestamp"],
                "Filename": filename,
                "Prediction": prediction.upper(),
                "Confidence": f"{confidence:.4f}",
                "Duration": f"{duration:.3f}",
            }
        ]
    )
    csv_data = csv_df.to_csv(index=False)
    with col2:
        st.download_button(
            label="📥 Export Report (CSV)",
            data=csv_data,
            file_name=f"report_{filename.split('.')[0]}.csv",
            mime="text/csv",
        )

    # 3. PDF Report Export
    pdf_bytes = generate_pdf_report(filename, prediction, confidence, duration, model_version)
    file_ext = "pdf" if has_fpdf else "txt"
    mime_type = "application/pdf" if has_fpdf else "text/plain"

    with col3:
        st.download_button(
            label=f"📥 Export Report ({file_ext.upper()})",
            data=pdf_bytes,
            file_name=f"report_{filename.split('.')[0]}.{file_ext}",
            mime=mime_type,
        )
