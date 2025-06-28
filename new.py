# main.py

import customtkinter as ctk
import os
import tkinter as tk

from encoder import encode_faces
from recognizer import FaceRecognizer
from test import capture_image

# ------------------- #
#  Helper Functions   #
# ------------------- #

def _open_camera_window(title: str, runner):
    cam_win = ctk.CTkToplevel()
    cam_win.title(title)
    cam_win.geometry("800x600")
    cam_win.resizable(False, False)
    runner(cam_win)

def take_image():
    _open_camera_window("Add Image – Camera", capture_image)

def mtcnn_recog():
    def _run(win):
        encode_faces()
        FaceRecognizer().recognize_from_video(win, 1)
    _open_camera_window("MTCNN Detection – Camera", _run)

def yolov8_recog():
    def _run(win):
        encode_faces()
        FaceRecognizer().recognize_from_video(win, 2)
    _open_camera_window("YOLOv8 Detection – Camera", _run)

def quit_app(window):
    try:
        window.destroy()
    except:
        pass
    os._exit(0)

# ------------------- #
#  Main GUI Function  #
# ------------------- #

def main():
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("green")

    root = ctk.CTk()
    root.title("Face Recognition App")
    root.geometry("450x400")
    root.minsize(450, 400)

    ctk.CTkLabel(
        root,
        text="🧠 Face Recognition System",
        font=ctk.CTkFont(size=22, weight="bold"),
    ).pack(pady=20)

    btn_cfg = dict(width=240, height=45, font=("Segoe UI", 14), corner_radius=8)

    ctk.CTkButton(root, text="➕  Add Image", command=take_image, **btn_cfg).pack(pady=8)
    ctk.CTkButton(root, text="📷  Start Detection (MTCNN)", command=mtcnn_recog, **btn_cfg).pack(pady=8)
    ctk.CTkButton(root, text="🎯  Start Detection (YOLOv8)", command=yolov8_recog, **btn_cfg).pack(pady=8)

    ctk.CTkButton(
        root,
        text="🚪  Quit App",
        command=lambda: quit_app(root),
        fg_color="#d9534f",
        hover_color="#c9302c",
        text_color="white",
        **btn_cfg
    ).pack(pady=(30, 10))

    add_utilities(root)
    root.mainloop()

# ------------------- #
#  Theme & Scale Bar  #
# ------------------- #

def add_utilities(root):
    util_frame = ctk.CTkFrame(root, fg_color="transparent")
    util_frame.pack(side="bottom", pady=(0, 15))

    mode_var = ctk.StringVar(value=ctk.get_appearance_mode().capitalize())
    ctk.CTkOptionMenu(
        util_frame,
        values=["Light", "Dark", "System"],
        variable=mode_var,
        command=ctk.set_appearance_mode,
        width=100
    ).pack(side="left", padx=10)

    scale_var = ctk.DoubleVar(value=1.0)
    ctk.CTkSlider(
        util_frame,
        from_=0.75,
        to=1.5,
        number_of_steps=6,
        variable=scale_var,
        command=ctk.set_widget_scaling,
        width=160
    ).pack(side="left", padx=10)

# ------------------- #
#  Entry Point        #
# ------------------- #

if __name__ == "__main__":
    main()
