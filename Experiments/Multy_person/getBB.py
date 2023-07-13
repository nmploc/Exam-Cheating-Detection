import cv2
from ultralytics import YOLO 
import torch

if torch.cuda.is_available():
    device = 0
else:
    device = 'cpu'
def get_bounding_boxes(frame):
    model = YOLO('yolov8n.pt')
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img, device = device)

    bounding_boxes = []  # Danh sách để lưu trữ tọa độ của bounding boxes

    # Find the bounding box coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            d = box.conf
            if model.names[int(c)] == 'person' and d > 0.8:
                b = box.xyxy[0]  # Get the bounding box coordinates
                x1= int(b[0])
                y1 = int(b[1])
                x2 = int(b[2])
                y2 = int(b[3])
                bounding_box = (x1, y1, x2, y2)
                bounding_boxes.append(bounding_box)

    # Return the list of bounding box coordinates
    return bounding_boxes
