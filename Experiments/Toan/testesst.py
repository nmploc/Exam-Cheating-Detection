import cv2 
from ultralytics import YOLO 
import time 
import numpy as np

#model = YOLO('yolov8m.pt')
model2 = YOLO('yolov8n-pose.pt')

# Load the image from file
image_path = 'jisoo_hit.jpg'
frame = cv2.imread(image_path)

# Perform object detection and pose estimation on the image
result2 = model2(frame, save=False, device='cpu')
keypoints = result2[0].keypoints

if keypoints is not None:
    keypoints_np = keypoints.cpu().numpy()
    print(keypoints_np)
    keypoints_np_reduce = np.squeeze(keypoints_np)
    print(keypoints_np.shape)
    print(keypoints_np_reduce.shape)

# Display the image with FPS information
'''
new_frame_time = time.time()
fps = 1 / (new_frame_time - prev_frame_time)
prev_frame_time = new_frame_time
fps = int(fps)
cv2.putText(frame, str(fps), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
'''


cv2.imshow('frame', frame)
cv2.waitKey(0)  # Wait until a key is pressed to close the window

cv2.destroyAllWindows()
