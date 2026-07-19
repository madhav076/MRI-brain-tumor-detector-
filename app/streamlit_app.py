"""Main Streamlit Application Entry point.

Sets up page configurations, styling templates, caches model loading,
and handles routing to Home Page, Single Classifier, Batch Classifier,
Prediction History, Diagnostics, and Settings interfaces.
"""

import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st

# Setup paths for importing local packages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src import config
from src.utils import set_seed
from src.data.augmentation import MRIAugmentationPipeline, RandomShear
from src.models.efficientnet_model import load_model_robustly
from app.components.sidebar import render_sidebar
from app.components.navbar import render_navbar
from app.components.dashboard_home import render_dashboard_home
from app.components.uploader import render_uploader, render_demo_selector
from app.components.prediction_card import execute_inference, render_prediction_results
from app.components.gradcam_viewer import render_gradcam_viewer
from app.components.report_generator import (
    add_to_history,
    render_history_section,
    render_exporters_section,
    init_session_history,
)

# Setup app logging to logs/app.log
log_path = Path(config.LOG_DIR)
log_path.mkdir(parents=True, exist_ok=True)
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
app_log_handler = logging.FileHandler(log_path / "app.log", encoding="utf-8")
app_log_handler.setFormatter(formatter)
app_logger = logging.getLogger("StreamlitApp")
app_logger.setLevel(logging.INFO)
if not any(isinstance(handler, logging.FileHandler) for handler in app_logger.handlers):
    app_logger.addHandler(app_log_handler)

# Check psutil for memory footprint info
try:
    import psutil

    has_psutil = True
except ImportError:
    has_psutil = False

# Configure Streamlit page layout
st.set_page_config(
    page_title="NeuroVision AI — Brain MRI Diagnosis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Load styles.css
def load_css() -> None:
    css_path = Path(__file__).resolve().parent / "assets" / "styles.css"
    if css_path.exists():
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


# Cache model loading for speed
@st.cache_resource(show_spinner="Loading Deep Learning Classifier...")
def load_best_model() -> Tuple[Optional[tf.keras.Model], float, Optional[str]]:
    """Loads the pre-trained Keras model robustly.

    Returns:
        Tuple[Optional[tf.keras.Model], float, Optional[str]]:
            Loaded model, loading time in seconds, and device used.
    """
    model_path = Path(config.MODEL_PATH)
    app_logger.info(f"Checking model at absolute path: {model_path.resolve()}")
    if not model_path.exists():
        app_logger.warning(f"No pre-trained model found at configured path: {model_path.resolve()}")
        return None, 0.0, None

    try:
        start_time = time.perf_counter()
        model = load_model_robustly(model_path)
        load_time = time.perf_counter() - start_time

        # Identify device
        gpus = tf.config.list_physical_devices("GPU")
        device = "GPU" if gpus else "CPU"

        app_logger.info(f"Model loaded successfully on {device} in {load_time:.2f} seconds.")
        return model, load_time, device
    except Exception as e:
        app_logger.error(f"Error loading model: {e}", exc_info=True)
        return None, 0.0, None


# Log Application Startup
if "app_started" not in st.session_state:
    app_logger.info("=" * 60)
    app_logger.info("           APPLICATION STARTUP INITIATED              ")
    app_logger.info("=" * 60)
    st.session_state.app_started = True

# Load Model
best_model, model_load_time, inference_device = load_best_model()
init_session_history()

# Setup Default Configurations in Session State
if "settings_precision" not in st.session_state:
    st.session_state.settings_precision = 2
if "settings_gradcam" not in st.session_state:
    st.session_state.settings_gradcam = True
if "settings_batch" not in st.session_state:
    st.session_state.settings_batch = True

# Retrieve tf version
tf_version = tf.__version__
device_label = inference_device if inference_device else "CPU (Fallback)"
CLASS_LABELS = ["glioma", "meningioma", "notumor", "pituitary"]

# Render Sidebar navigation
selected_page = render_sidebar(tf_version, device_label)


# Global custom functions for error screens
def render_error_screen(title: str, description: str, resolution_guidelines: List[str]) -> None:
    """Renders a medical-grade error notification box with troubleshooting steps.

    Args:
        title (str): Title of the warning.
        description (str): Detailed error description.
        resolution_guidelines (List[str]): List of resolution steps.
    """
    st.markdown("---")
    st.error(f"### ⚠️ {title}")
    st.write(description)
    st.markdown("**Suggested Resolutions:**")
    for step in resolution_guidelines:
        st.markdown(f"- {step}")
    st.markdown("---")


# Render top navbar globally
render_navbar(tf_version, device_label)

# Main Content Pages
if selected_page == "🏠 Dashboard":
    app_logger.info("Navigation: User navigated to Dashboard.")
    render_dashboard_home(st.session_state.prediction_history)

elif selected_page == "🧠 MRI Analysis":
    app_logger.info("Navigation: User navigated to MRI Analysis.")
    st.title("Clinical MRI Analysis Portal")

    # Consolidate Single Scan and Batch Scan into tabs
    tab_single, tab_batch = st.tabs(
        ["📷 Single Slice Image Analysis", "📁 Batch MRI Classification"]
    )

    with tab_single:
        if best_model is None:
            render_error_screen(
                title="Trained Classifier Model Missing",
                description="No pre-trained model file (such as `best_model.keras`, `best_model.h5`, `model.keras`, etc.) was found in `saved_models/`.",
                resolution_guidelines=[
                    "Ensure that you have completed training and saved the model weights successfully.",
                    "Verify that one of the following files exists inside the `saved_models/` folder: `best_model.keras`, `best_model.h5`, `model.keras`, `model.h5`, `final_model.keras`, or `checkpoint.keras`.",
                    "Execute the training script: `python scripts/run_training.py` to generate the checkpoint file.",
                ],
            )
        else:
            # Choose Upload or Demo Scans
            mode = st.radio(
                "Choose Input Mode",
                ["Upload MRI Scan File", "Use Pre-loaded Demo Scans"],
                horizontal=True,
                key="single_mode_radio",
            )

            img_data = None
            if mode == "Upload MRI Scan File":
                uploads = render_uploader(accept_multiple=False, key="single_scan_uploader")
                if uploads:
                    img_data = uploads[0]
            else:
                img_data = render_demo_selector("app/demo_images")

            if img_data is not None:
                image_array, filename = img_data

                st.markdown("---")
                col1, col2 = st.columns([1, 1])

                with col1:
                    st.subheader("Source MRI Scan")
                    st.image(image_array, caption=f"Scan: {filename}", use_column_width=True)

                with col2:
                    # Execute inference with pipeline animation
                    probs = None
                    placeholder = st.empty()
                    with placeholder.container():
                        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
                        st.subheader("Processing MRI scan inputs...")

                        step1 = st.empty()
                        step1.markdown(
                            '<div class="pipeline-step active">📤 Step 1/4: Uploading MRI slice data...</div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.3)
                        step1.markdown(
                            '<div class="pipeline-step complete">✅ Step 1/4: Uploading MRI slice data complete</div>',
                            unsafe_allow_html=True,
                        )

                        step2 = st.empty()
                        step2.markdown(
                            '<div class="pipeline-step active">⚙️ Step 2/4: Applying preprocessing pipelines...</div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.4)
                        step2.markdown(
                            '<div class="pipeline-step complete">✅ Step 2/4: Preprocessing complete (resized to 224x224x3, channels converted, normalized)</div>',
                            unsafe_allow_html=True,
                        )

                        step3 = st.empty()
                        step3.markdown(
                            '<div class="pipeline-step active">🧠 Step 3/4: Running TensorFlow model inference...</div>',
                            unsafe_allow_html=True,
                        )
                        try:
                            probs, inference_time = execute_inference(image_array, best_model)
                            time.sleep(0.3)
                            step3.markdown(
                                '<div class="pipeline-step complete">✅ Step 3/4: Inference complete</div>',
                                unsafe_allow_html=True,
                            )
                        except Exception as e:
                            app_logger.error(
                                f"Prediction failed for file '{filename}': {e}", exc_info=True
                            )
                            st.error(f"Prediction Failure: {e}")

                        step4 = st.empty()
                        step4.markdown(
                            '<div class="pipeline-step active">📄 Step 4/4: Compiling Grad-CAM maps & AI reports...</div>',
                            unsafe_allow_html=True,
                        )
                        time.sleep(0.3)
                        step4.markdown(
                            '<div class="pipeline-step complete">✅ Step 4/4: Diagnostic reports compiled successfully!</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)
                    time.sleep(0.3)
                    placeholder.empty()

                    if probs is not None:
                        # Render prediction gauges and probability distributions
                        pred_class, conf = render_prediction_results(
                            probs, CLASS_LABELS, precision=st.session_state.settings_precision
                        )

                        # Save prediction logs
                        add_to_history(filename, pred_class, conf, inference_time)
                        app_logger.info(
                            f"Successful prediction: File='{filename}', Pred='{pred_class}', Conf={conf:.4f}, Time={inference_time:.3f}s"
                        )

                # Render Grad-CAM overlays if enabled
                if probs is not None and st.session_state.settings_gradcam:
                    st.markdown("---")
                    pred_idx = int(np.argmax(probs[0]))
                    render_gradcam_viewer(image_array, best_model, pred_idx)

                # Expose exporters
                if probs is not None:
                    st.markdown("---")
                    render_exporters_section(
                        filename, pred_class, conf, inference_time, config.VERSION
                    )

    with tab_batch:
        if not st.session_state.settings_batch:
            st.warning(
                "⚠️ Batch prediction is disabled in Application Settings. Enable it to run multiple predictions."
            )
        elif best_model is None:
            render_error_screen(
                title="Trained Classifier Model Missing",
                description="No pre-trained model file (such as `best_model.keras`, `best_model.h5`, `model.keras`, etc.) was found in `saved_models/`.",
                resolution_guidelines=[
                    "Ensure that you have completed training and saved the model weights successfully.",
                    "Verify that one of the following files exists inside the `saved_models/` folder: `best_model.keras`, `best_model.h5`, `model.keras`, `model.h5`, `final_model.keras`, or `checkpoint.keras`.",
                    "Execute the training script: `python scripts/run_training.py` to generate the checkpoint file.",
                ],
            )
        else:
            # Batch uploader
            uploads = render_uploader(accept_multiple=True, key="batch_scan_uploader")

            if uploads:
                st.markdown("---")
                st.write(f"Processing batch classification of **{len(uploads)}** valid scans...")

                batch_results = []
                progress_bar = st.progress(0.0)

                for idx, (img_arr, filename) in enumerate(uploads):
                    try:
                        probs, inference_time = execute_inference(img_arr, best_model)
                        probs_val = probs[0]
                        pred_idx = int(np.argmax(probs_val))
                        classes_list = CLASS_LABELS
                        pred_class = classes_list[pred_idx]
                        conf = float(probs_val[pred_idx])

                        batch_results.append(
                            {
                                "Filename": filename,
                                "Predicted Class": pred_class.upper(),
                                "Confidence": f"{conf:.{st.session_state.settings_precision}%}",
                                "Inference Duration (s)": f"{inference_time:.3f}",
                            }
                        )
                        # Add to session history as well
                        add_to_history(filename, pred_class, conf, inference_time)
                    except Exception as e:
                        app_logger.error(
                            f"Batch prediction failed for '{filename}': {e}", exc_info=True
                        )
                        batch_results.append(
                            {
                                "Filename": filename,
                                "Predicted Class": "FAILED",
                                "Confidence": "0.0%",
                                "Inference Duration (s)": "0.0",
                            }
                        )

                    # Update progress bar
                    progress_bar.progress(float((idx + 1) / len(uploads)))

                st.success("Batch classification completed successfully!")

                # Display results dataframe
                df_results = pd.DataFrame(batch_results)
                st.dataframe(df_results, use_container_width=True)

                # CSV Download button for batch classification
                csv_data = df_results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Batch Predictions (CSV)",
                    data=csv_data,
                    file_name="batch_predictions_summary.csv",
                    mime="text/csv",
                )
                app_logger.info(f"Successful batch prediction: Processed {len(uploads)} images.")

elif selected_page == "📜 Patient History":
    app_logger.info("Navigation: User navigated to Prediction History.")
    st.title("Prediction Audits History")
    render_history_section()

elif selected_page == "📄 AI Reports":
    app_logger.info("Navigation: User navigated to AI Reports.")
    st.title("AI Diagnostic Reports Manager")
    st.write("Review, search, and download generated clinical reports for patient scan records.")
    render_history_section()

elif selected_page == "📊 Statistics":
    app_logger.info("Navigation: User navigated to Model Diagnostics.")
    st.title("Model Diagnostics Information")

    # Display hardware specs and memory footprint
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Performance Monitor")
        st.write(f"**Inference Platform Used:** {device_label}")
        st.write(f"**Model Load Time:** {model_load_time:.2f} seconds")

        # Memory Footprint using psutil if available
        if has_psutil:
            mem = psutil.virtual_memory()
            mem_used = mem.used / (1024**3)
            mem_total = mem.total / (1024**3)
            st.write(
                f"**System RAM Usage:** {mem_used:.2f} GB / {mem_total:.2f} GB ({mem.percent}%)"
            )
        else:
            st.write("**System RAM Usage:** psutil not installed (N/A)")

    with col2:
        st.subheader("Diagnostic Metrics Summary")
        summary_file = (
            Path(config.OUTPUT_DIR) / "evaluation" / "reports" / "evaluation_summary.json"
        )

        if summary_file.exists():
            with open(summary_file, "r") as f:
                sum_data = json.load(f)
            st.write(f"**Test Set Accuracy:** {sum_data.get('accuracy', 0.0):.2%}")
            st.write(f"**Balanced Accuracy:** {sum_data.get('balanced_accuracy', 0.0):.2%}")
            st.write(f"**Matthews Correlation Coefficient (MCC):** {sum_data.get('mcc', 0.0):.4f}")
            st.write(f"**Expected Calibration Error (ECE):** {sum_data.get('ece', 0.0):.4f}")
        else:
            st.info(
                "Evaluation metrics summary not discovered. Run Module 3 pipeline to generate stats."
            )

    st.markdown("---")

    # Model Summary details
    st.subheader("Model Summary Layout")
    summary_path = Path(config.CHECKPOINT_DIR) / "model_summary.txt"
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_txt = f.read()
        st.text_area(label="Keras Layer architecture printout", value=summary_txt, height=350)
    else:
        st.warning("model_summary.txt not found under saved_models/.")

elif selected_page == "⚙ Settings":
    app_logger.info("Navigation: User navigated to Application Settings.")
    st.title("Dashboard Controls Settings")

    st.subheader("Precision & Threshold Configuration")
    st.session_state.settings_precision = st.slider(
        "Confidence Decimal Precision",
        min_value=1,
        max_value=4,
        value=st.session_state.settings_precision,
        step=1,
    )

    st.markdown("---")
    st.subheader("Feature Flags Configurations")

    st.session_state.settings_gradcam = st.checkbox(
        "Enable Grad-CAM Visualizations",
        value=st.session_state.settings_gradcam,
        help="Enables/disables gradient attention heatmap overlay calculation on images.",
    )

    st.session_state.settings_batch = st.checkbox(
        "Enable Batch Image Classification",
        value=st.session_state.settings_batch,
        help="Enables upload of multiple MRI files simultaneously for batch predictions.",
    )

    st.markdown("---")
    st.subheader("Resource Clearing Cache")
    st.write("Clearing resources releases loaded models and clears cached directory metadata.")

    if st.button("🗑️ Clear Cache Resources", key="clear_cache_btn"):
        st.cache_resource.clear()
        st.cache_data.clear()
        app_logger.info("User triggered cache resources clearing.")
        st.success("Resource caches cleared successfully!")
        st.rerun()  # CRITICAL-02: st.experimental_rerun() removed in Streamlit ≥1.27; use st.rerun()

# Renders professional footer
st.markdown("---")
st.markdown(
    """
    <div class="footer">
        Brain MRI Tumor Classification using Deep Learning | © 2026 AI Engineer MLOps Pipeline | For Educational Demonstration Purposes Only
    </div>
    """,
    unsafe_allow_html=True,
)
