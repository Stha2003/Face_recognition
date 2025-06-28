# import cv2
# import pickle
# from keras_facenet import FaceNet
# from mtcnn.mtcnn import MTCNN
# from ultralytics import YOLO
# from face_utils import find_best_match
# import tkinter as tk
# from PIL import Image, ImageTk

# class FaceRecognizer:
#     def __init__(self, embedding_path="data_embeddings/embeddings.pkl"):
#         self.custom_detector = YOLO("custom/runs/detect/train2/weights/best.pt")
#         self.detector = MTCNN()
#         self.embedder = FaceNet()
#         with open(embedding_path, "rb") as f:
#             self.known_embeddings, self.known_names = pickle.load(f)

#     def recognize_from_video(self, window, model):
#         cap = cv2.VideoCapture(0)

#         label = tk.Label(window)
#         label.pack()
#         def update_frame():
#             ret, frame = cap.read()
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             match model: 
#                 case 1: 
#                     faces = self.detector.detect_faces(rgb)
#                     for face in faces: 
#                         x, y, w, h = face['box']
#                         face_img = rgb[y:y+h, x:x+w]
#                         try:
#                             embedding = self.embedder.embeddings([face_img])[0]
#                             name = find_best_match(embedding, self.known_embeddings, self.known_names)
#                             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                             cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
#                         except:
#                             continue
#                 case 2: 
#                     results = self.custom_detector.predict(rgb)
#                     for result in results:
#                         for box in result.boxes:
#                             x, y, x2, y2 = box.xyxy.cpu().numpy()[0]
#                             x, y, x2, y2 = int(x), int(y), int(x2), int(y2)
#                             w, h = x2 - x, y2 - y
#                             face_img = rgb[y:y2, x:x2]
#                             try:
#                                 embedding = self.embedder.embeddings([face_img])[0]
#                                 name = find_best_match(embedding, self.known_embeddings, self.known_names)
#                                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                                 cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
#                             except:
#                                 continue
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             img = Image.fromarray(frame)
#             imgtk = ImageTk.PhotoImage(image=img)
#             label.imgtk = imgtk
#             label.config(image=imgtk)
#             label.after(10, update_frame)

#         def on_close():
#             cap.release()
#             label.destroy()
#             exit_btn.destroy()
        
#         exit_btn = tk.Button(window, text= "Exit", command= on_close)
#         exit_btn.pack()
        

#         window.protocol("WM_DELETE_WINDOW", on_close)
#         update_frame()
#         window.mainloop()

            # cv2.imshow("Face Recognition", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        # cap.release()
        # cv2.destroyAllWindows()

    #  sumesh code: 
import cv2
import pickle
import tkinter as tk
from PIL import Image, ImageTk

from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet                 # FaceNet embedder
from ultralytics import YOLO                      # YOLOv8 detector

from face_utils import find_best_match            # your helper

# -----------------------------------------------------------
#  Face‑recognition class
# -----------------------------------------------------------
class FaceRecognizer:
    def __init__(self, embedding_path="data_embeddings/embeddings.pkl"):
        # ── models ────────────────────────────────────────────
        self.mtcnn = MTCNN()
        self.yolo  = YOLO("custom/runs/detect/train2/weights/best.pt")
        self.embedder = FaceNet()

        # ── known faces ───────────────────────────────────────
        with open(embedding_path, "rb") as f:
            self.known_embeddings, self.known_names = pickle.load(f)

        # ── runtime vars filled later ─────────────────────────
        self.cap     = None      # cv2.VideoCapture
        self.window  = None      # Tk / CTk window
        self.label   = None      # Tk label holding the frame
        self.model_id = 1        # 1 = MTCNN, 2 = YOLOv8

    # --------------------------------------------------------
    #  Public entry‑point
    # --------------------------------------------------------
    def recognize_from_video(self, window, model_id: int):
        """Start webcam recognition inside the given Tk window."""
        self.window   = window
        self.model_id = model_id
        self.cap      = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        self.label = tk.Label(self.window)
        self.label.pack()

        self.window.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_frame()                            # kick‑start loop

    # --------------------------------------------------------
    #  Frame loop
    # --------------------------------------------------------
    def _update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.window.after(10, self._update_frame)
            return

        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = self._detect_faces(rgb)                # [(x1,y1,x2,y2), …]

        if not boxes:                                  # ── No faces
            self._draw_status(frame, "No face")
        else:                                          # ── At least one face
            for (x1, y1, x2, y2) in boxes:
                face_crop = rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                try:
                    emb = self.embedder.embeddings([face_crop])[0]
                except Exception:
                    continue                           # bad crop → skip

                name = find_best_match(
                    emb, self.known_embeddings, self.known_names
                )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        self._show(frame)
        self.window.after(10, self._update_frame)      # schedule next frame

    # --------------------------------------------------------
    #  Face detectors
    # --------------------------------------------------------
    def _detect_faces(self, rgb_img):
        if self.model_id == 1:                         # ── MTCNN
            mtcnn_faces = self.mtcnn.detect_faces(rgb_img)
            boxes = []
            for det in mtcnn_faces:
                x, y, w, h = det["box"]
                x, y = max(0, x), max(0, y)            # clamp to img bounds
                boxes.append((x, y, x + w, y + h))
            return boxes

        if self.model_id == 2:                         # ── YOLOv8
            results = self.yolo.predict(rgb_img, verbose=False)
            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
                    boxes.append((x1, y1, x2, y2))
            return boxes

        return []                                      # fallback (shouldn’t happen)

    # --------------------------------------------------------
    #  Drawing / display helpers
    # --------------------------------------------------------
    @staticmethod
    def _draw_status(frame, text):
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def _show(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        img     = Image.fromarray(rgb_img)
        imgtk   = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk                       # prevent GC
        self.label.configure(image=imgtk)

    # --------------------------------------------------------
    #  Clean shutdown
    # --------------------------------------------------------
    def _on_close(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.window.destroy()
