import cv2 
from ultralytics import YOLO 

model = YOLO('yolov8m.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    result = model(frame, save = False)
    frame = result[0].plot()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()