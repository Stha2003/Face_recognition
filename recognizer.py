# recognizer_streamlit.py
import streamlit as st
import cv2, pickle
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from ultralytics import YOLO
from face_utils import find_best_match

class FaceRecognizer:
    def __init__(self, emb_path="data_embeddings/embeddings.pkl"):
        self.mtcnn  = MTCNN()
        self.yolo   = YOLO("custom/runs/detect/train2/weights/best.pt")
        self.embed  = FaceNet()

        with open(emb_path, "rb") as f:
            self.known_emb, self.known_names = pickle.load(f)

    # ------------- Streamlit live loop -------------
    def recognize_from_video_streamlit(self, model_id=1):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
            return

        st_frame = st.empty()
        stop = st.button("🛑 Stop Recognition")

        while cap.isOpened() and not stop:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = self._detect(rgb, model_id)

            for (x1, y1, x2, y2) in boxes:
                face = rgb[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                try:
                    emb = self.embed.embeddings([face])[0]
                except Exception:
                    continue
                name = find_best_match(emb, self.known_emb, self.known_names)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, name, (x1, max(0, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
            stop = st.button("🛑 Stop Recognition")

        cap.release()
        st.success("Recognition stopped.")

    # ---------- Helpers ----------
    def _detect(self, rgb, model_id):
        if model_id == 1:              # MTCNN
            det = self.mtcnn.detect_faces(rgb)
            return [(max(0,x), max(0,y), x+w, y+h) for d in det
                    for x,y,w,h in [d['box']]]
        else:                          # YOLOv8
            res = self.yolo.predict(rgb, verbose=False)
            out=[]
            for r in res:
                for b in r.boxes:
                    x1,y1,x2,y2 = map(int, b.xyxy.cpu().numpy()[0])
                    out.append((x1,y1,x2,y2))
            return out
