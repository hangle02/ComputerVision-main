import cv2
import numpy as np
import os
import glob

def detect_full_road(img):
    """
    Hàm xử lý tìm 2 lề đường và đánh dấu tâm cho 1 bức ảnh.
    Trả về ảnh đã được vẽ thêm đường line và chấm đỏ.
    """
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # 1. Tạo ROI (Mặt nạ chữ nhật nửa dưới ảnh)
    mask = np.zeros_like(edges)
    cv2.rectangle(mask, (0, int(height * 0.55)), (width, height), 255, -1)
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
            if abs(slope) < 0.3 or abs(slope) > 5:
                continue
                
            intercept = y1 - slope * x1
            x_bottom = (height - intercept) / slope
            
            # Phân loại lề trái / phải dựa vào độ dốc
            if slope < 0:
                left_road_edges.append((x_bottom, slope, intercept))
            else:
                right_road_edges.append((x_bottom, slope, intercept))

    # 3. Lọc 2 lề ngoài cùng và vẽ
    if len(left_road_edges) > 0 and len(right_road_edges) > 0:
        # Đường bên trái xa nhất
        left_road_edges.sort(key=lambda x: x[0])
        _, m_left, b_left = left_road_edges[0]
        
        # Đường bên phải xa nhất
        right_road_edges.sort(key=lambda x: x[0], reverse=True)
        _, m_right, b_right = right_road_edges[0]
        
        # Vẽ 2 lề
        y1_draw = height
        y2_draw = int(height * 0.6)
        
        x1_left = int((y1_draw - b_left) / m_left)
        x2_left = int((y2_draw - b_left) / m_left)
        cv2.line(img, (x1_left, y1_draw), (x2_left, y2_draw), (0, 255, 0), 4)
        
        x1_right = int((y1_draw - b_right) / m_right)
        x2_right = int((y2_draw - b_right) / m_right)
        cv2.line(img, (x1_right, y1_draw), (x2_right, y2_draw), (255, 0, 0), 4)
        
        # Đánh dấu tâm
        target_y = int(height * 0.8) 
        x_left_edge = int((target_y - b_left) / m_left)
        x_right_edge = int((target_y - b_right) / m_right)
        
        mid_x = int((x_left_edge + x_right_edge) / 2)
        
        cv2.circle(img, (mid_x, target_y), 10, (0, 0, 255), -1)
        cv2.putText(img, "Full Road Center", (mid_x - 80, target_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return img

def process_folder(input_folder, output_folder):
    """
    Đọc tất cả ảnh trong thư mục input, xử lý và lưu vào output
    """
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Lấy danh sách các file ảnh (hỗ trợ jpg, png, bmp...)
    image_paths = glob.glob(os.path.join(input_folder, '*.*'))
    
    # Lọc ra các file ảnh hợp lệ (tránh đọc nhầm file txt, zip...)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = [p for p in image_paths if p.lower().endswith(valid_extensions)]

    if len(image_paths) == 0:
        print(f"Không tìm thấy ảnh nào trong thư mục '{input_folder}'")
        return

    print(f"Bắt đầu xử lý {len(image_paths)} ảnh...")

    # Duyệt qua từng ảnh và xử lý
    for i, img_path in enumerate(image_paths):
        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"Lỗi đọc ảnh: {img_path}")
            continue
            
        # Gọi hàm xử lý
        processed_img = detect_full_road(img)
        
        # Tạo đường dẫn lưu file
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_folder, f"out_{filename}")
        
        # Lưu ảnh
        cv2.imwrite(output_path, processed_img)
        print(f"[{i+1}/{len(image_paths)}] Đã lưu: {output_path}")

    print("Hoàn tất! Hãy kiểm tra thư mục đầu ra.")

# ==========================================
# CHẠY CHƯƠNG TRÌNH
# ==========================================
if __name__ == '__main__':
    # Tên thư mục chứa ảnh gốc
    INPUT_DIR = 'input_images'
    # Tên thư mục lưu ảnh sau khi xử lý
    OUTPUT_DIR = 'output_images'
    
    # Tạo sẵn thư mục input nếu bạn chưa tạo để tránh báo lỗi
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR)
        print(f"Đã tạo thư mục '{INPUT_DIR}'. Hãy copy ảnh vào đây và chạy lại code.")
    else:
        process_folder(INPUT_DIR, OUTPUT_DIR)