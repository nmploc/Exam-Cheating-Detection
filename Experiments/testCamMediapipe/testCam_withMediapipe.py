import cv2
import mediapipe as mp
from getBB import get_bounding_box
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Load the pose data from the CSV file
pose_data = []
with open('landmask.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        pose_data.append(row)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to read frame from camera.")
            break

        # Get the bounding box coordinates
        bounding_box = get_bounding_box()

        if bounding_box:
            top, left, bottom, right = bounding_box

            # Crop the frame within the bounding box
            cropped_image = image[top:bottom, left:right]

            # Convert the cropped image to RGB
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            # Process the cropped image with Mediapipe Pose
            results = pose.process(cropped_image_rgb)

            # Draw the pose landmarks on the cropped image
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image=cropped_image,
                    landmark_list=results.pose_landmarks,
                    connections=mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

            # Display the cropped image with pose landmarks
            cv2.imshow('Camera', cropped_image)
        else:
            cv2.putText(image, 'No person detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Camera', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
