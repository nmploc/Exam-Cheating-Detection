from ultralytics import YOLO
import cv2
import time

def get_bounding_box():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Capture a frame
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    # Find the bounding box coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            if model.names[int(c)] == 'person':
                b = box.xyxy[0]  # Get the bounding box coordinates
                top = int(b[0])
                left = int(b[1])
                bottom = int(b[2])
                right = int(b[3])

    # Return the bounding box coordinates
    return top, left, bottom, right

