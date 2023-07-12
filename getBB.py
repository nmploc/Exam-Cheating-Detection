from ultralytics import YOLO
import cv2
import time

def get_bounding_box(frame):
<<<<<<< HEAD
    model = YOLO('yolov8m.pt')

    # Capture a frame

=======
    model = YOLO('yolov8n.pt')
>>>>>>> 1f7ecf8825421a0ed4e7f4395f59fe6d7e13f2df
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img, device = 0)
    x1 = 0 
    y1 = 0 
    x2 = 0 
    y2 = 0 
    # Find the bounding box coordinates
    for r in results:
        boxes = r.boxes
        for box in boxes:
            c = box.cls
            if model.names[int(c)] == 'person':
                b = box.xyxy[0]  # Get the bounding box coordinates
                x1 = int(b[0]) #x1
                y1 = int(b[1]) #y1
                x2 = int(b[2]) #x2
                y2 = int(b[3]) #y2

    # Return the bounding box coordinates
    return x1, y1, x2, y2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        x1, y1, x2, y2 = get_bounding_box(frame)
        print(str(x1) + " " + str(y1) + " " + str(x2) +" "+ str(y2))
        roi = frame[y1:y2, x1:x2]
        cv2.imshow("Roi",roi)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Release video capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
