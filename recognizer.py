import cv2
import pickle
from keras_facenet import FaceNet
from mtcnn.mtcnn import MTCNN
from ultralytics import YOLO
from face_utils import find_best_match
import tkinter as tk
from PIL import Image, ImageTk

class FaceRecognizer:
    def __init__(self, embedding_path="data_embeddings/embeddings.pkl"):
        self.custom_detector = YOLO("custom/runs/detect/train2/weights/best.pt")
        self.detector = MTCNN()
        self.embedder = FaceNet()
        with open(embedding_path, "rb") as f:
            self.known_embeddings, self.known_names = pickle.load(f)

    def recognize_from_video(self, window, model):
        cap = cv2.VideoCapture(0)

        label = tk.Label(window)
        label.pack()
        def update_frame():
            ret, frame = cap.read()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            match model: 
                case 1: 
                    faces = self.detector.detect_faces(rgb)
                    for face in faces: 
                        x, y, w, h = face['box']
                        face_img = rgb[y:y+h, x:x+w]
                        try:
                            embedding = self.embedder.embeddings([face_img])[0]
                        except:
                            continue
                case 2: 
                    results = self.custom_detector.predict(rgb)
                    for result in results:
                        for box in result.boxes:
                            x, y, x2, y2 = box.xyxy.cpu().numpy()[0]
                            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
                            w, h = x2 - x, y2 - y
                            face_img = rgb[y:y2, x:x2]
                            try:
                                embedding = self.embedder.embeddings([face_img])[0]
                            except:
                                continue
            name = find_best_match(embedding, self.known_embeddings, self.known_names)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)
            label.after(10, update_frame)

        def on_close():
            cap.release()
            label.destroy()
            exit_btn.destroy()
        
        exit_btn = tk.Button(window, text= "Exit", command= on_close)
        exit_btn.pack()
        

        window.protocol("WM_DELETE_WINDOW", on_close)
        update_frame()
        window.mainloop()

            # cv2.imshow("Face Recognition", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # cap.release()
        # cv2.destroyAllWindows()