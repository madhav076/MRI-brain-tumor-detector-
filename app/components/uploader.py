"""Streamlit App Image Uploader and Validator Component.

Handles drag-and-drop file inputs, performs binary/format checks,
and provides a interface to load scans from the local demo directory.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Setup logging
logger = logging.getLogger(__name__)

def validate_uploaded_image(file_bytes: bytes, filename: str) -> Optional[np.ndarray]:
    """Validates the uploaded file binary contents.

    Checks that the image is not empty, is readable by OpenCV, and has valid dimensions.

    Args:
        file_bytes (bytes): Binary file array.
        filename (str): Name of the file.

    Returns:
        Optional[np.ndarray]: Decoded OpenCV image array (grayscale or BGR),
                              or None if validation failed.
    """
    if len(file_bytes) == 0:
        logger.error(f"Validation failed: Uploaded file '{filename}' is empty (0 bytes).")
        st.error(f"Uploaded file **{filename}** is empty (0 bytes). Please upload a valid image.")
        return None

    try:
        # Convert binary bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        # Decode using OpenCV
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error(f"Validation failed: OpenCV failed to decode '{filename}' headers.")
            st.error(f"Uploaded file **{filename}** is corrupted or not a valid image format.")
            return None
            
        return img
    except Exception as e:
        logger.error(f"Unexpected error validating image '{filename}': {e}", exc_info=True)
        st.error(f"Error reading **{filename}**: {e}")
        return None

def render_uploader(accept_multiple: bool = False, key: str = "mri_uploader") -> List[Tuple[np.ndarray, str]]:
    """Renders the file uploader widget and returns validated image arrays.

    Args:
        accept_multiple (bool): True to support batch uploads.
        key (str): Unique widget key identifier.

    Returns:
        List[Tuple[np.ndarray, str]]: List of tuples containing (image_array, filename).
    """
    st.subheader("Image Upload Portal")
    uploaded_files = st.file_uploader(
        "Upload Brain MRI slice scans (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=accept_multiple,
        key=key
    )

    if not uploaded_files:
        return []

    valid_images = []
    
    # Normalize list wrapper for single vs multiple files
    files_list = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]

    for f in files_list:
        file_bytes = f.read()
        # Reset file seek pointer
        f.seek(0)
        
        img_arr = validate_uploaded_image(file_bytes, f.name)
        if img_arr is not None:
            valid_images.append((img_arr, f.name))

    return valid_images

def render_demo_selector(demo_dir_path: str = "app/demo_images") -> Optional[Tuple[np.ndarray, str]]:
    """Scans and renders a dropdown selector for local demo scans.

    Args:
        demo_dir_path (str): Folder path holding demo scans.

    Returns:
        Optional[Tuple[np.ndarray, str]]: Tuple of (image_array, filename) or None.
    """
    demo_dir = Path(demo_dir_path)
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    # Scan files
    demo_files = [
        f for f in demo_dir.iterdir()
        if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not demo_files:
        st.info(
            f"💡 **Demo Mode**: To enable demo testing without manual uploads, "
            f"place test MRI scans in the folder: `{demo_dir.resolve()}`"
        )
        return None

    st.subheader("Try Demo Scans")
    filenames = [f.name for f in demo_files]
    selected_name = st.selectbox(
        "Select a pre-loaded MRI scan to test the pipeline:",
        options=["-- Choose a Demo Scan --"] + filenames
    )

    if selected_name == "-- Choose a Demo Scan --":
        return None

    # Load selected demo scan
    selected_file = demo_dir / selected_name
    img = cv2.imread(str(selected_file.resolve()), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        st.error(f"Failed to load demo image: {selected_name}")
        return None
        
    return img, selected_name
