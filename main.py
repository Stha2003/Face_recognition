from encoder import encode_faces
from recognizer import FaceRecognizer
from test import capture_image

if __name__ == "__main__":

    # Hmm for now this will work.
    while True:
        action = int(input("What do you want to do? \n 1) Add Image \n 2) Detect \n 3) Exit\n"))
        match action:
            case 1: 
                capture_image()
            case 2: 
                # First encode faces (run once)
                encode_faces()

                # Then start real-time recognition
                recognizer = FaceRecognizer()
                recognizer.recognize_from_video()
            case 3: 
                exit()