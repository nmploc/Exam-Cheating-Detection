import cv2 
from ultralytics import YOLO 
import time 
import numpy as np
#model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8n-pose.pt')
cap = cv2.VideoCapture(0)

prev_frame_time = 0
new_frame_time = 0

while True:
    frame = cv2.imread('D:\Exam-Cheating-Detection\Experiments\TUNG\jisoo_hit.jpg') 
    
    #bounding_boxes = model(frame)
    #print(bounding_boxes)
    #result = model(frame, save = False)q
    result2 = model2(frame, save = False, device = 'cpu')
    keypoints = result2[0].keypoints
    keypoints_np = keypoints.numpy()
    #print(keypoints_np)
    '''
    keypoints_np_reduce = np.squeeze(keypoints_np)
    print(keypoints_np.shape)
    print(keypoints_np_reduce.shape)
    print("in ra toa do cac keypoint")
    for person in keypoints_np:
        for keypoint in person:
            x, y, confidence = keypoint
            if confidence > 0.3: 
                print(f"Keypoint: X={x}, Y={y}")
    #frame = result[0].plot()
    frame = result2[0].plot()
    '''
    kp_coordinates = keypoints_np.xy[0]
    #print(keypoints_np.shape)
    x = kp_coordinates[1][0]
    y = kp_coordinates[1][1]

    cv2.circle(frame, (int(x), int(y)), 5, (255,0,0), thickness=5)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()