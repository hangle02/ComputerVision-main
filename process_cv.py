import cv2
import numpy as np
import time

class ImageProcessor:
    def __init__(self):
        # Biến dùng cho tính năng Auto-Scan (Theo dõi chuyển động)
        self.prev_corners = None
        self.stable_counter = 0
        self.STABLE_THRESHOLD = 20  # Sai số pixel cho phép khi tay rung nhẹ
        self.FRAMES_TO_LOCK = 3     # Số frame đứng im liên tiếp để chốt chụp

    def order_points(self, pts):
        """ Sắp xếp 4 điểm theo thứ tự: Trái-Trên, Phải-Trên, Phải-Dưới, Trái-Dưới """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process_frame(self, frame, step='all'):
        start_time = time.time()
        results = {
            "status": "success", 
            "message": "Processed successfully",
            "is_stable": False  # Cờ báo cho giao diện web tự động chụp
        }
        orig = frame.copy()
        
        # =========================================================
        # GIAI ĐOẠN 1: TẨY CHỮ (MAGIC ERASER) & TÌM VIỀN THÔNG MINH
        # =========================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Morphological Close (Kernel lớn): "Nuốt chửng" toàn bộ chữ viết đen 
        # và dòng kẻ ngang, biến tờ giấy thành một khối trắng tinh mờ ảo.
        kernel_eraser = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blank_paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_eraser)

        # 2. Làm mờ nhẹ để xóa nhiễu vân gỗ mặt bàn
        blurred = cv2.GaussianBlur(blank_paper, (5, 5), 0)

       # 3. THEO ĐÚNG Ý TƯỞNG CỦA BẠN: ÉP THRESHOLD XUỐNG CỰC THẤP
        # Bỏ Auto-Canny, fix cứng ngưỡng (20, 60) để Canny nhạy cảm hơn, 
        # bắt được cả những ranh giới lờ mờ nhất giữa giấy trắng và bàn gỗ.
        edged = cv2.Canny(blurred, 20, 60)

        # 4. "Đổ bê tông" hàn gắn các khoảng trống khổng lồ
        # Ảnh của bạn nét đứt rất xa, nên ta kết hợp cả MORPH_CLOSE và DILATE 
        # với kernel to (11x11) để ép các đường thẳng đứt quãng nối lại với nhau.
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_bridge)
        edged = cv2.dilate(edged, kernel_bridge, iterations=1)

        if step == 'edges':
            process_time_ms = (time.time() - start_time) * 1000
            return cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR), results, process_time_ms

        # =========================================================
        # GIAI ĐOẠN 2: TÌM CONTOURS & VƯỢT QUA GÁY LÒ XO
        # =========================================================
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_area = orig.shape[0] * orig.shape[1]
        doc_cnt = None

        for c in contours:
            # Bỏ qua các contour rác có diện tích quá nhỏ
            if cv2.contourArea(c) < 0.05 * image_area:
                continue

            # Bao lồi (Convex Hull): Bọc màng nilon phẳng qua gáy lò xo nhấp nhô
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            
            # Thử nghiệm các mức nới lỏng để ép hình dạng về 4 đỉnh
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                
                if len(approx) == 4:
                    # Đảm bảo tứ giác là hình lồi (loại bỏ lỗi móp méo do bóng đèn)
                    if cv2.isContourConvex(approx):
                        doc_cnt = approx
                        break 
            
            if doc_cnt is not None:
                break

        # =========================================================
        # LOGIC THEO DÕI CHUYỂN ĐỘNG (AUTO-CAPTURE)
        # =========================================================
        box_color = (0, 0, 255) # Mặc định viền màu Đỏ (Đang dịch chuyển)

        if doc_cnt is not None:
            current_corners = self.order_points(doc_cnt.reshape(4, 2))
            
            if self.prev_corners is not None:
                # Tính khoảng cách Euclid giữa tọa độ góc cũ và mới
                dist = np.max(np.linalg.norm(current_corners - self.prev_corners, axis=1))
                if dist < self.STABLE_THRESHOLD:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0 
            else:
                self.stable_counter = 0
                
            self.prev_corners = current_corners
            
            # Nếu cầm tay đủ vững trong 3 frames -> Đổi viền Xanh Lá và bật cờ
            if self.stable_counter >= self.FRAMES_TO_LOCK:
                results["is_stable"] = True
                box_color = (0, 255, 0)
        else:
            self.stable_counter = 0
            self.prev_corners = None

        if doc_cnt is None:
            results["status"] = "failed"
            results["message"] = "No document found in the frame."
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        if step == 'contours':
            # Vẽ viền thực tế (vàng) và vẽ 4 góc chốt hạ (Đỏ/Xanh)
            cv2.drawContours(orig, [c], -1, (0, 255, 255), 2)  
            cv2.drawContours(orig, [doc_cnt], -1, box_color, 3) 
            for point in doc_cnt:
                cv2.circle(orig, tuple(point[0]), 8, box_color, -1)
                
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        # =========================================================
        # GIAI ĐOẠN 3: ĐẠI SỐ TUYẾN TÍNH (HOMOGRAPHY)
        # =========================================================
        pts = doc_cnt.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        results["homography_matrix"] = M.tolist() 
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        if step == 'warp':
            process_time_ms = (time.time() - start_time) * 1000
            return warped, results, process_time_ms

        # =========================================================
        # GIAI ĐOẠN 4: LỌC NHIỄU & LÀM SẠCH VĂN BẢN (POST-PROCESSING)
        # =========================================================
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # Denoising: Triệt tiêu các bóng râm gắt trước khi binarize
        denoised = cv2.fastNlMeansDenoising(warped_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # Adaptive Threshold: Tẩy trắng nền giấy, xóa dòng kẻ vở (blockSize = 51)
        scanned = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            51,   
            15    
        )
        
        # Xóa các chấm đen nhiễu hạt nhỏ lấm tấm
        scanned = cv2.medianBlur(scanned, 3)

        # Bóp vùng màu trắng lại để chữ viết tay màu đen hiển thị đậm đà hơn
        kernel_text = np.ones((2, 2), np.uint8)
        scanned = cv2.erode(scanned, kernel_text, iterations=1)

        process_time_ms = (time.time() - start_time) * 1000
        scanned_bgr = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)
        
        return scanned_bgr, results, process_time_ms