from encoder import encode_faces
from recognizer import FaceRecognizer

if __name__ == "__main__":
    # First encode faces (run once)
    encode_faces()

    # Then start real-time recognition
    recognizer = FaceRecognizer()
    recognizer.recognize_from_video()
