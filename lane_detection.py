import cv2
import numpy as np

img = cv2.imread('test3.jpg')
if img is None:
    print("Không tìm thấy ảnh.")
    exit()

height, width = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

# ==========================================
# 1. FIX LỖI ROI Ở ĐÂY: DÙNG HÌNH CHỮ NHẬT 
# ==========================================
mask = np.zeros_like(edges)
# Lấy toàn bộ nửa dưới bức ảnh (Từ 55% chiều cao đổ xuống đáy)
# Như vậy sẽ không lề đường nào bị cắt mất
cv2.rectangle(mask, (0, int(height * 0.55)), (width, height), 255, -1)

# Áp dụng mặt nạ
masked_edges = cv2.bitwise_and(edges, mask)

# 2. Hough Transform
lines = cv2.HoughLinesP(
    masked_edges, 1, np.pi/180, 40, minLineLength=40, maxLineGap=100
)

left_road_edges = []
right_road_edges = []

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
            
        slope = (y2 - y1) / (x2 - x1)
        
        # Bỏ qua các đường nằm ngang (nhiễu)
        if abs(slope) < 0.3 or abs(slope) > 5:
            continue
            
        intercept = y1 - slope * x1
        x_bottom = (height - intercept) / slope
        
        # Đường dốc âm là lề trái, dốc dương là lề phải
        if slope < 0:
            left_road_edges.append((x_bottom, slope, intercept))
        else:
            right_road_edges.append((x_bottom, slope, intercept))

# 3. Lọc 2 lề ngoài cùng
if len(left_road_edges) > 0 and len(right_road_edges) > 0:
    
    # Sắp xếp để lấy đường xa nhất về BÊN TRÁI (x_bottom NHỎ NHẤT)
    left_road_edges.sort(key=lambda x: x[0])
    outer_left = left_road_edges[0] 
    
    # Sắp xếp để lấy đường xa nhất về BÊN PHẢI (x_bottom LỚN NHẤT)
    right_road_edges.sort(key=lambda x: x[0], reverse=True)
    outer_right = right_road_edges[0]
    
    _, m_left, b_left = outer_left
    _, m_right, b_right = outer_right
    
    # --- VẼ 2 LỀ ĐƯỜNG MÀU XANH LÁ VÀ XANH DƯƠNG ---
    y1_draw = height
    y2_draw = int(height * 0.6)
    
    x1_left = int((y1_draw - b_left) / m_left)
    x2_left = int((y2_draw - b_left) / m_left)
    cv2.line(img, (x1_left, y1_draw), (x2_left, y2_draw), (0, 255, 0), 4)
    
    x1_right = int((y1_draw - b_right) / m_right)
    x2_right = int((y2_draw - b_right) / m_right)
    cv2.line(img, (x1_right, y1_draw), (x2_right, y2_draw), (255, 0, 0), 4)
    
    # --- ĐÁNH DẤU ROAD CENTER ---
    target_y = int(height * 0.8) 
    
    x_left_edge = int((target_y - b_left) / m_left)
    x_right_edge = int((target_y - b_right) / m_right)
    
    mid_x = int((x_left_edge + x_right_edge) / 2)
    
    cv2.circle(img, (mid_x, target_y), 10, (0, 0, 255), -1)
    cv2.putText(img, "Full Road Center", (mid_x - 80, target_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# Show kết quả
cv2.imshow('Full Road Center Fixed', img)
cv2.waitKey(0)
cv2.destroyAllWindows()