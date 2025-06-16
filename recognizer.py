import cv2
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from .face_utils import find_best_match

class FaceRecognizer:
    def __init__(self, embedding_path="data/embeddings.pkl"):
        self.detector = MTCNN()
        self.embedder = FaceNet()
        with open(embedding_path, "rb") as f:
            self.known_embeddings, self.known_names = pickle.load(f)

    def recognize_from_video(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector.detect_faces(rgb)

            for face in faces:
                x, y, w, h = face['box']
                face_img = rgb[y:y+h, x:x+w]
                try:
                    embedding = self.embedder.embeddings([face_img])[0]
                except:
                    continue
                name = find_best_match(embedding, self.known_embeddings, self.known_names)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()