import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
import pickle

embedder = FaceNet()
detector = MTCNN()

data_dir = "dataset"
embeddings = []
labels = []

for person_name in os.listdir(data_dir):
    person_dir = os.path.join(data_dir, person_name)
    for image_name in os.listdir(person_dir):
        image_path = os.path.join(person_dir, image_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        faces = detector.detect_faces(rgb_image)
        if faces:
            x, y, w, h = faces[0]['box']
            face_img = rgb_image[y:y+h, x:x+w]
            embedding = embedder.embeddings([face_img])[0]
            embeddings.append(embedding)
            labels.append(person_name)

with open("embeddings/embeddings.pkl", "wb") as f:
    pickle.dump((embeddings, labels), f)
