from ultralytics import YOLO
import cv2
import numpy as np
import time
from ultralytics.yolo.utils.plotting import Annotator

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, frame = cap.read()
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)
    for r in results:
        
        annotator = Annotator(frame)
        
        boxes = r.boxes
        for box in boxes:
            
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            top = int(b[0])
            left = int(b[1])
            bottom = int(b[2])
            right = int(b[3])
            c = box.cls
            if model.names[int(c)] == 'person':
                annotator.box_label(b, model.names[int(c)])
                # Crop the region of interest
                roi = frame[top:bottom, left:right]
                # Display the cropped ROI
                cv2.imshow('ROI', roi)
            
          
    frame = annotator.result()  
    cv2.imshow('YOLO V8 Detection', frame)     
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

