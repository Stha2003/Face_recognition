# streamlit_app.py
import streamlit as st
from encoder import encode_faces
from recognizer_streamlit import FaceRecognizer
from test_streamlit import capture_image

# ------------------------- UI setup -------------------------
st.set_page_config(page_title="Face Recognition System", layout="centered")
st.title("🧠 Face Recognition System")

# Style helpers – mimic your CTk theme
primary_color = "#00a86b"          # green
danger_color  = "#d9534f"

def primary_btn(label, key=None):
    return st.button(label, key=key, use_container_width=True)

def danger_btn(label, key=None):
    return st.button(label, key=key, use_container_width=True,
                     type="secondary", help="Close the tab to exit")

st.markdown("---")

# ------------------------- Actions --------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    if primary_btn("➕  Add Image", key="add_img"):
        capture_image()

with col2:
    if danger_btn("🚪  Quit App", key="quit"):
        st.warning("Close this browser tab to quit the app.")

st.markdown("## Real‑time Detection")

if st.button("📷  Start Detection (MTCNN)"):
    encode_faces()
    FaceRecognizer().recognize_from_video_streamlit(model_id=1)

if st.button("🎯  Start Detection (YOLOv8)"):
    encode_faces()
    FaceRecognizer().recognize_from_video_streamlit(model_id=2)
