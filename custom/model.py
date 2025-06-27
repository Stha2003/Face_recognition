from ultralytics import YOLO


model = YOLO("yolov8n.pt")  # Pretrained weights

# model training 
model.train(data="data/data.yaml", epochs=5, imgsz = 512)


results = model.predict(source="Austin.jpg", save=True, save_txt=True)

# 7. Extract Bounding Box Coordinates
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # pixel coordinates
        w = x2 - x1
        h = y2 - y1
        print(f"Lower-left: ({int(x1)}, {int(y2)}), Width: {int(w)}, Height: {int(h)}")