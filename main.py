# from encoder import encode_faces
# from recognizer import FaceRecognizer
# from test import capture_image

# if __name__ == "__main__":

#     # Hmm for now this will work.
#     while True:
#         action = int(input("What do you want to do? \n 1) Add Image \n 2) Detect \n 3) Exit\n"))
#         match action:
#             case 1: 
#                 capture_image()
#             case 2: 
#                 # First encode faces (run once)
#                 encode_faces()

#                 # Then start real-time recognition
#                 recognizer = FaceRecognizer()
#                 recognizer.recognize_from_video()
#             case 3: 
#                 exit()


import glob
import tkinter as tk

from idna import encode
from encoder import encode_faces
from recognizer import FaceRecognizer
from test import capture_image

window = tk.Tk()

def take_image():
    global window
    capture_image(window)

def mtcnn_recog():
    global window
    encode_faces()
    # Then start real-time recognition
    recognizer = FaceRecognizer()
    recognizer.recognize_from_video(window, 1)

def yolov8_recog():
    global window
    encode_faces()
    # Then start real-time recognition
    recognizer = FaceRecognizer()
    recognizer.recognize_from_video(window, 2)


def main():
    global window
    window.title("Face recognition")
    window.geometry("900x700")

    

    # Add Image
    image_button = tk.Button(
        window, 
        text= "Add Image",
        command= take_image,
        background= "Green",
        pady= 10,
        padx = 10
    )
    image_button.pack()

    # detect using mtcnn
    mtcnn_button = tk.Button(
        window, 
        text= "Start Detection (MTCNN)",
        command= mtcnn_recog,
        background= "Green",
        pady= 10,
        padx = 10
    )
    mtcnn_button.pack()
    
    # detect using yolov8 model
    mtcnn_button = tk.Button(
        window, 
        text= "Start Detection (Yolov8)",
        command= yolov8_recog,
        background= "Green",
        pady= 10,
        padx = 10
    )
    mtcnn_button.pack()

    # Exit button 
    exit_button = tk.Button(
        window, 
        text= "Exit",
        command= exit,
        background= "red",
        pady= 10,
        padx = 10
    )
    exit_button.pack()
    

    window.mainloop()
main()