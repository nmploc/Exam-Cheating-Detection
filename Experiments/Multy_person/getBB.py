import cv2
from ultralytics import YOLO 
def get_bounding_boxes():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    # Capture a frame
    _, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)

    bounding_boxes = []  # Danh sách để lưu trữ tọa độ của bounding boxes

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
                bounding_box = (top, left, bottom, right)
                bounding_boxes.append(bounding_box)

    # Return the list of bounding box coordinates
    return bounding_boxes
