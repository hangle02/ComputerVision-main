import cv2
import numpy as np
import time

class ImageProcessor:
    def __init__(self):
        pass

    def order_points(self, pts):
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
        results = {"status": "success", "message": "Processed successfully"}
        orig = frame.copy()
        
        # =========================================================
        # GIAI ĐOẠN 1: MÔ PHỎNG AI MASKING CỦA FAIRSCAN
        # Bỏ hoàn toàn Canny. Tạo một khối Mask (Mặt nạ) trắng đen.
        # =========================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Làm mờ cực mạnh (kernel 15x15) để nhòe chữ viết và vân gỗ
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)

        # 2. Tách nền bằng Otsu Thresholding (Tự động tìm ngưỡng sáng tốt nhất)
        _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Phép đóng (Morphological Closing) - Hàn gắn các vết rách trên mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Đôi khi Otsu bị ngược (Giấy đen, bàn trắng). Check 4 góc để đảo lại nếu cần:
        corners = [mask[0,0], mask[-1,0], mask[0,-1], mask[-1,-1]]
        if sum(1 for c in corners if c > 127) >= 2:
            mask = cv2.bitwise_not(mask)

        # Nếu người dùng chọn xem bước 1, trả về cái Mask này (ảnh trắng đen)
        if step == 'edges':
            process_time_ms = (time.time() - start_time) * 1000
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), results, process_time_ms

        # =========================================================
        # GIAI ĐOẠN 2: TÌM CONTOURS & VƯỢT QUA GÁY LÒ XO
        # =========================================================
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_area = orig.shape[0] * orig.shape[1]
        doc_cnt = None

        for c in contours:
            if cv2.contourArea(c) < 0.1 * image_area:
                continue

            # CHIẾN THUẬT FAIRSCAN: Bọc "Bao lồi" (Convex Hull) quanh contour
            # Điều này giúp lờ đi hoàn toàn các vết lõm của gáy lò xo!
            hull = cv2.convexHull(c)

            peri = cv2.arcLength(hull, True)
            # Áp dụng approxPolyDP trên cái Bao Lồi phẳng phiu đó
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4:
                    doc_cnt = approx
                    break 
            
            if doc_cnt is not None:
                break

        if doc_cnt is None:
            results["status"] = "failed"
            results["message"] = "No document found in the frame."
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        if step == 'contours':
            # Vẽ Contour gốc (vàng) và Bao lồi (xanh lá) để thầy cô thấy độ ảo diệu
            cv2.drawContours(orig, [c], -1, (0, 255, 255), 2)  # Vàng: Viền thật
            cv2.drawContours(orig, [doc_cnt], -1, (0, 255, 0), 3) # Xanh: Khung 4 góc
            for point in doc_cnt:
                cv2.circle(orig, tuple(point[0]), 8, (255, 0, 0), -1)
                
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        # =========================================================
        # GIAI ĐOẠN 3: HOMOGRAPHY (Như cũ)
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
        # GIAI ĐOẠN 3: LỌC NHIỄU VÀ LÀM SẠCH VĂN BẢN (HẬU KỲ)
        # Bắt đầu xử lý trên biến 'warped' (mảnh giấy đã được bẻ phẳng)
        # =========================================================
        
        # 1. Chuyển mảnh giấy sang ảnh xám
        warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        
        # 2. Khử nhiễu nền nâng cao (Fast Non-Local Means Denoising)
        # Bước này cực kỳ xịn để làm mờ các vân giấy và dòng kẻ mờ 
        # TRƯỚC KHI cắt ngưỡng. (Tùy chọn: nếu app chạy chậm thì có thể bỏ bước này)
        denoised = cv2.fastNlMeansDenoising(warped_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 3. Tẩy trắng nền, giữ nét chữ (Adaptive Thresholding)
        # BÍ QUYẾT: 
        # - Block Size = 51 (Số cực lớn, bắt buộc là số lẻ). Nó ép thuật toán lờ đi 
        #   các dòng kẻ ngang và nhìn vào tổng thể ánh sáng cả trang giấy.
        # - C = 15 (Hằng số trừ). Kéo các vệt xám mờ (vân giấy) thành màu trắng tinh.
        scanned = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            51,   # Tăng thông số này lên 71, 91 nếu giấy vẫn còn dòng kẻ
            15    # Tăng thông số này lên 20 nếu ảnh vẫn còn lốm đốm xám
        )
        
        # 4. Xóa nhiễu hạt (Salt & Pepper Noise)
        # Các lốm đốm đen li ti còn sót lại sẽ bị Median Blur "nhai" sạch.
        # Dùng kernel 3x3 để không làm nhòe chữ viết tay.
        scanned = cv2.medianBlur(scanned, 3)

        # 5. Phục hồi độ đậm của nét chữ (Morphology Erode)
        # Vì chữ viết tay thường bị mỏng đi sau khi Threshold, ta dùng Erode 
        # để "bóp" vùng màu trắng lại, khiến vùng màu đen (chữ) đậm và rõ ràng hơn.
        kernel_text = np.ones((2, 2), np.uint8)
        scanned = cv2.erode(scanned, kernel_text, iterations=1)

        # Trả về kết quả
        process_time_ms = (time.time() - start_time) * 1000
        scanned_bgr = cv2.cvtColor(scanned, cv2.COLOR_GRAY2BGR)
        
        return scanned_bgr, results, process_time_ms