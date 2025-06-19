# app/encoder.py
import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pickle

embedder = FaceNet()
detector = MTCNN()

def encode_faces(dataset_dir="data", save_path="data_embeddings/embeddings.pkl"):
    embeddings = []
    labels = []

    for image_name in os.listdir(dataset_dir):
        image_path = os.path.join(dataset_dir, image_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb_image)
        if faces:
            x, y, w, h = faces[0]['box']
            face_img = rgb_image[y:y+h, x:x+w]
            embedding = embedder.embeddings([face_img])[0]
            name = os.path.splitext(image_name)[0]  # Use file name as label
            embeddings.append(embedding)
            labels.append(name)

    with open(save_path, "wb") as f:
        pickle.dump((embeddings, labels), f)
    print("[INFO] Embeddings saved to", save_path)