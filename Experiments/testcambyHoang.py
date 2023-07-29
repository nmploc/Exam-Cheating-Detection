import cv2 
from ultralytics import YOLO 
import time 

model2 = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while True:
    ret, frame = cap.read() 
    
    result2 = model2(frame, save=False, device='cpu')
    
    # Lấy tọa độ keypoint ở mũi
    if len(result2[0].keypoints) > 0:
        nose_keypoint = result2[0].keypoints[0]['keypoints'][0:2]  # Tọa độ x, y của mũi
        print("Tọa độ mũi: ", nose_keypoint)
    
    frame = result2[0].plot()
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    cv2.putText(frame, "FPS: " + str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
