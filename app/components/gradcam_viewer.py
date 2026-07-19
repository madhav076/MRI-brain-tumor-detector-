"""Streamlit App Grad-CAM Viewer Component.

Generates localized prediction overlays using GradCAMExplainer,
provides blending opacity controls, and supports visual downloads.
"""

import logging

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf

from src import config
from src.evaluation.explainability.gradcam import GradCAMExplainer

# Setup logging
logger = logging.getLogger(__name__)


def render_gradcam_viewer(
    image_array: np.ndarray, model: tf.keras.Model, pred_class_idx: int
) -> None:
    """Generates and renders the Grad-CAM visualization interface.

    Allows user to slide opacity levels and download the combined image maps.

    Args:
        image_array (np.ndarray): Original image array.
        model (tf.keras.Model): Loaded Keras model.
        pred_class_idx (int): Predicted category index target for Grad-CAM.
    """
    st.markdown('<div class="premium-card">', unsafe_allow_html=True)
    # 1. Overlay opacity slider control
    alpha = st.slider(
        "Heatmap Overlay Opacity (Alpha)",
        min_value=0.1,
        max_value=0.9,
        value=0.4,
        step=0.05,
        key="gradcam_alpha_slider",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        from src.data.preprocessing import preprocess_single_image

        # Preprocess using identical pipeline (RGB, resized, normalized)
        img_tensor = preprocess_single_image(
            image_array, target_size=config.IMAGE_SIZE, normalize_method="minmax_01", is_bgr=True
        )

        # Extract uint8 RGB representation for overlay blending
        img_resized = (img_tensor.numpy() * 255.0).astype(np.uint8)

        # Add batch dim for explain()
        img_tensor_batch = tf.expand_dims(img_tensor, axis=0)

        # Generate Heatmap
        explainer = GradCAMExplainer()
        heatmap = explainer.explain(model, img_tensor_batch, pred_class_idx)

        # Apply overlay blending
        # overlaid image output is RGB representation
        overlaid = explainer.overlay_heatmap(img_resized, heatmap, alpha=alpha)

        st.markdown('<div class="premium-card" style="padding: 20px;">', unsafe_allow_html=True)
        # Render 3 columns side-by-side
        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(img_resized, caption="Original MRI", use_column_width=True)

        with col2:
            st.image(heatmap, caption="Attention Heatmap", use_column_width=True, clamp=True)

        with col3:
            st.image(overlaid, caption="Blended Overlay", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Renders overlaid download button
        # Convert RGB to BGR for OpenCV encoding
        overlaid_bgr = cv2.cvtColor(overlaid, cv2.COLOR_RGB2BGR)
        success, encoded_img = cv2.imencode(".png", overlaid_bgr)

        if success:
            img_bytes = encoded_img.tobytes()
            st.download_button(
                label="📥 Download Overlaid Grad-CAM scan (PNG)",
                data=img_bytes,
                file_name=f"gradcam_overlay_class_{pred_class_idx}.png",
                mime="image/png",
            )
            logger.info("Successfully encoded and generated download bytes for Grad-CAM overlay.")
        else:
            logger.error("Failed to encode Grad-CAM overlay image to PNG.")

    except Exception as e:
        logger.error(f"Failed to generate/render Grad-CAM outputs: {e}", exc_info=True)
        st.error(f"Interpretability Engine error: {e}. Please try checking model details.")
