import cv2
import numpy as np
import time

class ImageProcessor:
    def __init__(self):
        # Bạn có thể khởi tạo các tham số mặc định ở đây nếu cần
        pass

    def order_points(self, pts):
        """
        Hàm sắp xếp 4 điểm của tứ giác theo thứ tự: 
        Top-Left, Top-Right, Bottom-Right, Bottom-Left
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def process_frame(self, frame, step='all'):
        """
        Hàm xử lý ảnh chính.
        step có thể là: 'edges', 'contours', 'warp', hoặc 'all'
        """
        start_time = time.time()
        results = {"status": "success", "message": "Processed successfully"}
        
        # Tạo bản sao của ảnh gốc để vẽ đè lên
        orig = frame.copy()
        
        # ---------------------------------------------------------
        # BƯỚC 1 & 2: Xử lý Computer Vision (Tiền xử lý & Tìm biên)
        # ---------------------------------------------------------
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 75, 200)

        if step == 'edges':
            process_time_ms = (time.time() - start_time) * 1000
            # Trả về ảnh xám (cần convert lại BGR để hiển thị web thống nhất)
            return cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR), results, process_time_ms

        # ---------------------------------------------------------
        # BƯỚC 3: Tìm viền tài liệu (Contours)
        # ---------------------------------------------------------
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        doc_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # Nếu đa giác xấp xỉ có 4 đỉnh, ta giả định đó là tờ giấy
            if len(approx) == 4:
                doc_cnt = approx
                break

        # Nếu không tìm thấy tờ giấy, trả về ảnh gốc kèm thông báo
        if doc_cnt is None:
            results["status"] = "failed"
            results["message"] = "No document found in the frame."
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        if step == 'contours':
            # Vẽ đường viền màu xanh lá cây dày 2px
            cv2.drawContours(orig, [doc_cnt], -1, (0, 255, 0), 2)
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        # ---------------------------------------------------------
        # BƯỚC 4: Linear Algebra (Tính toán Homography & Cắt phẳng)
        # ---------------------------------------------------------
        # Định hình lại mảng tọa độ
        pts = doc_cnt.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

        # Tính toán chiều rộng và chiều cao tối đa của tài liệu mới
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        # Tọa độ đích của hình chữ nhật phẳng
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Core Math: Giải hệ phương trình tìm Ma trận Homography (3x3)
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Lưu ma trận M vào results để gửi lên Frontend show cho giảng viên xem
        results["homography_matrix"] = M.tolist() 

        # Nhân ma trận để bẻ phẳng ảnh
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        if step == 'warp':
            process_time_ms = (time.time() - start_time) * 1000
            return warped, results, process_time_ms

        # ---------------------------------------------------------
        # BƯỚC 5: Hậu xử lý (Làm trắng nền đen chữ như máy Scan)
        # ---------------------------------------------------------
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Sử dụng Adaptive Thresholding
        scanned = cv2.adaptiveThreshold(warped_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)

        process_time_ms = (time.time() - start_time) * 1000
        # Convert lại BGR để hiển thị
        scanned_bgr = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)
        
        return scanned_bgr, results, process_time_ms