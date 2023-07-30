import cv2
import numpy as np

# Tạo một mảng numpy chứa các tọa độ xy
points = np.array([[
                    [681.48, 528.99],
                    [567.5, 520.31],
                    ]], dtype=np.int32)

# Đọc bức ảnh 'jisoo.jpg'
img = cv2.imread('jisoo.jpg')

# Vẽ các điểm chấm màu khác nhau lên các tọa độ này
for point in points[0]:
    x = int(point[0])
    y = int(point[1])
    cv2.circle(img,(x,y), radius=10, color=np.random.randint(0,255), thickness=-1)

# Lưu kết quả ra một folder tên là 'results'
cv2.imwrite('results/image.jpg', img)
