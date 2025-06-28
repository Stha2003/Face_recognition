# test_streamlit.py
import streamlit as st
import cv2, os, datetime
import numpy as np
from PIL import Image

def capture_image():
    """Open webcam in Streamlit, let user capture, save to /data."""
    img_file = st.camera_input("Take a picture")
    if img_file:
        img = Image.open(img_file)
        img_array = np.array(img)      # RGB
        os.makedirs("data", exist_ok=True)
        fname = datetime.datetime.now().strftime("data/%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(fname, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved as **{fname}**")
