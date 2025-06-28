# test.py

import cv2
import os
import datetime
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import simpledialog

def capture_image(window):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    label = tk.Label(window)
    label.pack()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.config(image=imgtk)
            label.current_frame = frame
        label.after(10, update_frame)

    def save_frame():
        if hasattr(label, "current_frame"):
            name = simpledialog.askstring("Save Image", "Enter image name:")
            if name:
                os.makedirs("data", exist_ok=True)
                path = f"data/{name}.jpg"
                cv2.imwrite(path, label.current_frame)
                print(f"✅ Image saved as: {path}")
        close_window()

    def close_window():
        cap.release()
        window.destroy()

    # Buttons
    save_btn = tk.Button(window, text="💾 Save Image", command=save_frame)
    save_btn.pack(pady=10)

    quit_btn = tk.Button(window, text="❌ Cancel", command=close_window)
    quit_btn.pack(pady=5)

    window.protocol("WM_DELETE_WINDOW", close_window)
    update_frame()
