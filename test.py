import cv2

def capture_image():
    cap = cv2.VideoCapture(0)  # Open the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    print("Press 's' to save image or 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            image_name = input("Enter Name: ")
            cv2.imwrite(f"data/{image_name}.jpg", frame)
            print(f"Image saved as {image_name}.jpg")
            break
        elif key == ord('q'):
            print("Quitting without saving.")
            break

    cap.release()
    cv2.destroyAllWindows()
