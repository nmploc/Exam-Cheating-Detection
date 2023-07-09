from getBB import get_bounding_box
import cv2
import mediapipe 
# Gọi hàm để lấy bounding box coordinates
top, left, bottom, right = get_bounding_box()

# In tọa độ bounding box
print("Bounding box coordinates:")
print(f"Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")

# Hiển thị bounding box trên frame
cap = cv2.VideoCapture(0)
while True:
    _, frame = cap.read()
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
