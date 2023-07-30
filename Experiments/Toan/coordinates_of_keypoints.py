import cv2 
from ultralytics import YOLO 
import time 
import numpy as np

model = YOLO('yolov8n-pose.pt')


results = model('download.jpg', show=True, stream=True)
# frame = results
# result2 = model(frame, save = False, device = 'cpu')
for result in results:
    keypoints = result[0].keypoints.numpy()
    for keypoint in keypoints:
        print(keypoint)

    # boxes = result[0].boxes.numpy()
    # for box in box:
    #     print("class", box.cls)
    #     print("xyxy", box.xyxy)
    #     print("conf", box.conf)
