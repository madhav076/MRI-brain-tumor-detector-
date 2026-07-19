"""Streamlit App Sidebar Component.

Renders project metadata, navigation options, active framework versions,
and the medical disclaimer notice.
"""

import streamlit as st
from pathlib import Path
from src import config


def render_sidebar(tf_version: str, device: str) -> str:
    """Renders the navigation sidebar and returns the selected page option.

    Args:
        tf_version (str): Active TensorFlow engine version.
        device (str): Inference device platform (CPU/GPU).

    Returns:
        str: Selected page name.
    """
    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-bottom: 20px; padding-top: 10px;'>
            <h2 style='color: #2563EB; font-weight: 800; font-size: 1.6rem; margin-bottom: 4px; letter-spacing: -0.025em;'>NeuroVision AI</h2>
            <p style='color: #6B7280; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700; line-height: 1.2;'>AI Powered Brain MRI Diagnosis Platform</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # Navigation Radio Buttons
    page = st.sidebar.radio(
        "Application Navigation",
        options=[
            "🏠 Dashboard",
            "🧠 MRI Analysis",
            "📜 Patient History",
            "📄 AI Reports",
            "📊 Statistics",
            "⚙ Settings",
        ],
    )

    st.sidebar.markdown("---")

    # System metadata summary
    st.sidebar.markdown("### System Diagnostics")
    st.sidebar.markdown(
        f"""
        <div style='font-size: 0.85rem; color: #9CA3AF; line-height: 1.6; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); padding: 12px; border-radius: 10px;'>
            <span style='color: #6B7280;'>Base Model:</span> {config.MODEL_NAME}<br>
            <span style='color: #6B7280;'>Version:</span> {config.VERSION}<br>
            <span style='color: #6B7280;'>Framework:</span> TensorFlow v{tf_version}<br>
            <span style='color: #6B7280;'>Device:</span> {device}<br>
            <span style='color: #6B7280;'>Categories:</span> {config.NUM_CLASSES} classes
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("---")

    # Global Medical Disclaimer Notice
    st.sidebar.warning(
        "⚠️ **MEDICAL DISCLAIMER**\n\n"
        "This application is an educational AI demonstration project. "
        "It is NOT a medical diagnosis tool and must NOT be used for clinical decisions."
    )

    st.sidebar.markdown(
        """
        <div style='text-align: center; margin-top: 20px; font-size: 0.75rem; color: #4B5563;'>
            Developed by Madhav
        </div>
        """,
        unsafe_allow_html=True,
    )

    return page
