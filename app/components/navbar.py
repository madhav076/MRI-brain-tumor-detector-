"""Streamlit App Top Navbar Component.

Renders clinician profiles, notifications, and model diagnostic details.
"""

import streamlit as st
import tensorflow as tf
from src import config

def render_navbar(tf_version: str, device: str) -> None:
    """Renders the professional enterprise top navigation navbar.

    Args:
        tf_version (str): TensorFlow package version.
        device (str): System inference device target platform.
    """
    navbar_html = f"""
    <div class="navbar-container">
        <div class="navbar-brand">
            🧠 NeuroVision AI <span style="font-size: 0.8rem; font-weight: normal; color: #9CA3AF; margin-left: 8px;">v{config.VERSION}</span>
        </div>
        <div class="navbar-meta">
            <div style="font-size: 0.85rem; color: #9CA3AF; background: rgba(31, 41, 55, 0.5); border: 1px solid rgba(255, 255, 255, 0.05); padding: 6px 14px; border-radius: 8px; display: flex; align-items: center; gap: 8px;">
                <span class="status-dot"></span> Engine: TF v{tf_version} ({device})
            </div>
            <div style="display: flex; align-items: center; gap: 8px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); padding: 6px 14px; border-radius: 8px;">
                <span style="width: 8px; height: 8px; border-radius: 50%; background-color: #10B981; display: inline-block; box-shadow: 0 0 8px #10B981;"></span>
                <span style="font-size: 0.85rem; font-weight: 600; color: #10B981;">AI Ready</span>
            </div>
        </div>
    </div>
    """
    st.markdown(navbar_html, unsafe_allow_html=True)
