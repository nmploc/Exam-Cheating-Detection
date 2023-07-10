import cv2
import csv
import mediapipe as mp
from getBB import get_bounding_box 

# Function to read landmarks from CSV file
def read_landmarks_from_csv(file_path):
    landmarks = []
    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header row
        for row in csv_reader:
            landmarks.append(row)
    return landmarks

# Function to display landmarks on frame
def draw_landmarks(frame, landmarks):
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

# Main function
def main():
    # Load YOLO model and get bounding box coordinates from getBB.py
    top, left, bottom, right = get_bounding_box()

    # Load Mediapipe solution for face landmarks
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh()

    # Read landmarks from CSV file
    landmarks = read_landmarks_from_csv('landmarks.csv')

    # Open video capture
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        # Read frame
        _, frame = cap.read()

        # Crop frame using bounding box coordinates
        cropped_frame = frame[top:bottom, left:right]

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe FaceMesh
        results = mp_face_mesh.process(frame_rgb)

        # Extract landmarks from Mediapipe results
        if results.multi_face_landmarks:
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            extracted_landmarks = []
            # Extract desired landmarks
            x1 = int(face_landmarks.landmark[landmarks[0]].x * cropped_frame.shape[1])
            y1 = int(face_landmarks.landmark[landmarks[0]].y * cropped_frame.shape[0])
            extracted_landmarks.append([x1, y1])

            x2 = int(face_landmarks.landmark[landmarks[1]].x * cropped_frame.shape[1])
            y2 = int(face_landmarks.landmark[landmarks[1]].y * cropped_frame.shape[0])
            extracted_landmarks.append([x2, y2])

# Tiếp tục viết từng điểm còn lại tương tự


            # Draw landmarks on frame
            draw_landmarks(cropped_frame, extracted_landmarks)

        # Display frame
        cv2.imshow("Frame", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
