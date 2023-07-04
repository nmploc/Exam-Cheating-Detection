import cv2 
from ultralytics import YOLO 

#model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8m-pose.pt')
 
model2.to('cuda')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() 
    fps =  int(cap.get(cv2.CAP_PROP_FPS))
    #bounding_boxes = model(frame)
    #print(bounding_boxes)
    #result = model(frame, save = False)
    result2 = model2.predict(frame, save = False, device = 0)
    #print(result[0])
    print(result2[0])
    #frame = result[0].plot()
    frame = result2[0].plot()
    cv2.putText(frame, str(fps), (50,50), cv2.FONT_HERSHEY_COMPLEX , 1,  (0,255,0), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()