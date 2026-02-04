import cv2
import numpy as np

# 1. Đọc ảnh và tiền xử lý
img = cv2.imread('test3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 2. Áp dụng HoughLinesP
# tham số: (ảnh_edge, độ_chính_xác_rho, độ_chính_xác_theta, ngưỡng_bầu_chọn, độ_dài_min, khoảng_hở_max)
lines = cv2.HoughLinesP(
    edges, 
    rho=1,                # Độ chính xác khoảng cách (thường là 1 pixel)
    theta=np.pi/180,      # Độ chính xác góc (thường là 1 độ)
    threshold=100,        # Số điểm tối thiểu trên đường thẳng để được chấp nhận
    minLineLength=100,    # Độ dài tối thiểu của đường thẳng (ngắn hơn sẽ bị bỏ)
    maxLineGap=10         # Khoảng cách tối đa giữa các đoạn đứt nét để coi là 1 đường
)

# 3. Vẽ đường thẳng
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 4. Hiển thị
cv2.imshow('HoughLinesP Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()