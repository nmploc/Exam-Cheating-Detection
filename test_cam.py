import cv2 
from ultralytics import YOLO 

model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8m-pose.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    result = model(frame, save = False)
    result2 = model2(frame, save = False)
    print(result[0])
    print(result2[0])
    frame = result[0].plot()
    frame = result2[0].plot()
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()