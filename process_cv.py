import cv2
import numpy as np
import time

class ImageProcessor:
    def __init__(self):
        # Variables used for Auto-Scan (Motion tracking)
        self.prev_corners = None
        self.stable_counter = 0
        self.STABLE_THRESHOLD = 20  # Allowed pixel tolerance for slight hand shaking
        self.FRAMES_TO_LOCK = 3     # Number of consecutive stable frames to lock the capture

    def order_points(self, pts):
        """ Order 4 points in the sequence: Top-Left, Top-Right, Bottom-Right, Bottom-Left """
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
            "is_stable": False
        }
        orig = frame.copy()
        
        # STAGE 1: Edges
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_eraser = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blank_paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_eraser)
        blurred = cv2.GaussianBlur(blank_paper, (5, 5), 0)
        edged = cv2.Canny(blurred, 20, 60)
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_bridge)
        edged = cv2.dilate(edged, kernel_bridge, iterations=1)

        if step == 'edges':
            process_time_ms = (time.time() - start_time) * 1000
            return cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR), results, process_time_ms

        # STAGE 2: Contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        image_area = orig.shape[0] * orig.shape[1]
        doc_cnt = None

        for c in contours:
            if cv2.contourArea(c) < 0.05 * image_area:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    doc_cnt = approx
                    break 
            if doc_cnt is not None:
                break

        box_color = (0, 0, 255)
        if doc_cnt is not None:
            current_corners = self.order_points(doc_cnt.reshape(4, 2))
            if self.prev_corners is not None:
                dist = np.max(np.linalg.norm(current_corners - self.prev_corners, axis=1))
                if dist < self.STABLE_THRESHOLD:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0 
            else:
                self.stable_counter = 0
            self.prev_corners = current_corners
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
            cv2.drawContours(orig, [c], -1, (0, 255, 255), 2)  
            cv2.drawContours(orig, [doc_cnt], -1, box_color, 3) 
            for point in doc_cnt:
                cv2.circle(orig, tuple(point[0]), 8, box_color, -1)
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        # STAGE 3: Warp
        pts = doc_cnt.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        results["homography_matrix"] = M.tolist() 
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

        if step == 'warp':
            process_time_ms = (time.time() - start_time) * 1000
            return warped, results, process_time_ms

        # STAGE 4: All (Sharpening)
        scanned_bgr = cv2.convertScaleAbs(warped, alpha=1.2, beta=25)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        scanned_bgr = cv2.filter2D(scanned_bgr, -1, sharpen_kernel)

        process_time_ms = (time.time() - start_time) * 1000
        return scanned_bgr, results, process_time_ms

    def process_all_stages(self, frame):
        """ Thực hiện toàn bộ Pipeline, tính thời gian độc lập của từng Step """
        results = {"status": "success", "message": "Processed successfully", "is_stable": False}
        stages_images = {}
        stages_times = {}
        orig = frame.copy()

        # Step 1: Edges
        t0 = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel_eraser = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blank_paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_eraser)
        blurred = cv2.GaussianBlur(blank_paper, (5, 5), 0)
        edged = cv2.Canny(blurred, 20, 60)
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_bridge)
        edged = cv2.dilate(edged, kernel_bridge, iterations=1)
        stages_times['edges'] = (time.time() - t0) * 1000
        stages_images['edges'] = cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR)

        # Step 2: Contours
        t1 = time.time()
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        image_area = orig.shape[0] * orig.shape[1]
        doc_cnt = None

        for c in contours:
            if cv2.contourArea(c) < 0.05 * image_area:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    doc_cnt = approx
                    break 
            if doc_cnt is not None:
                break

        box_color = (0, 0, 255)
        if doc_cnt is not None:
            current_corners = self.order_points(doc_cnt.reshape(4, 2))
            if self.prev_corners is not None:
                dist = np.max(np.linalg.norm(current_corners - self.prev_corners, axis=1))
                if dist < self.STABLE_THRESHOLD:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0 
            else:
                self.stable_counter = 0
            self.prev_corners = current_corners
            if self.stable_counter >= self.FRAMES_TO_LOCK:
                results["is_stable"] = True
                box_color = (0, 255, 0)
        else:
            self.stable_counter = 0
            self.prev_corners = None

        contour_img = orig.copy()
        if doc_cnt is not None:
            cv2.drawContours(contour_img, [c], -1, (0, 255, 255), 2)  
            cv2.drawContours(contour_img, [doc_cnt], -1, box_color, 3) 
            for point in doc_cnt:
                cv2.circle(contour_img, tuple(point[0]), 8, box_color, -1)
        stages_times['contours'] = (time.time() - t1) * 1000
        stages_images['contours'] = contour_img

        if doc_cnt is None:
            results["status"] = "failed"
            results["message"] = "No document found."
            stages_times['warp'] = 0
            stages_times['all'] = 0
            stages_images['warp'] = orig.copy()
            stages_images['all'] = orig.copy()
            return stages_images, stages_times, results

        # Step 3: Warp
        t2 = time.time()
        pts = doc_cnt.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        results["homography_matrix"] = M.tolist() 
        warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
        stages_times['warp'] = (time.time() - t2) * 1000
        stages_images['warp'] = warped.copy()

        # Step 4: Enhanced Scan (Final)
        t3 = time.time()
        scanned_bgr = cv2.convertScaleAbs(warped, alpha=1.2, beta=25)
        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        scanned_bgr = cv2.filter2D(scanned_bgr, -1, sharpen_kernel)
        stages_times['all'] = (time.time() - t3) * 1000
        stages_images['all'] = scanned_bgr

        return stages_images, stages_times, results