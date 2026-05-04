import cv2
import numpy as np
import time

class ImageProcessor:
    def __init__(self):
        # Variables used for the Auto-Scan feature (Motion tracking)
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
            "is_stable": False  # Flag to signal the web interface to auto-capture
        }
        orig = frame.copy()
        
        # =========================================================
        # STAGE 1: TEXT REMOVAL (MAGIC ERASER) & SMART EDGE DETECTION
        # =========================================================
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Morphological Close (Large Kernel): "Swallows" all black text 
        # and horizontal lines, turning the paper into a blurry white block.
        kernel_eraser = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        blank_paper = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_eraser)

        # 2. Slight blur to remove wood grain noise from the table
        blurred = cv2.GaussianBlur(blank_paper, (5, 5), 0)

        # 3. AS PER YOUR IDEA: FORCE THRESHOLD EXTREMELY LOW
        # Discard Auto-Canny, hardcode threshold to (20, 60) to make Canny more sensitive, 
        # capturing even the faintest boundaries between white paper and wooden table.
        edged = cv2.Canny(blurred, 20, 60)

        # 4. "Pour concrete" to bridge giant gaps
        # Your image has widely disconnected lines, so we combine MORPH_CLOSE and DILATE 
        # with a large kernel (11x11) to force broken lines to connect together.
        kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel_bridge)
        edged = cv2.dilate(edged, kernel_bridge, iterations=1)

        if step == 'edges':
            process_time_ms = (time.time() - start_time) * 1000
            return cv2.cvtColor(edged, cv2.COLOR_GRAY2BGR), results, process_time_ms

        # =========================================================
        # STAGE 2: FIND CONTOURS & OVERCOME SPIRAL BINDING
        # =========================================================
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        image_area = orig.shape[0] * orig.shape[1]
        doc_cnt = None

        for c in contours:
            # Ignore noise contours with an area that is too small
            if cv2.contourArea(c) < 0.05 * image_area:
                continue

            # Convex Hull: Wrap a flat "plastic wrap" over the bumpy spiral binding
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            
            # Experiment with loosening levels to force the shape into 4 vertices
            for eps in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(hull, eps * peri, True)
                
                if len(approx) == 4:
                    # Ensure the quadrilateral is convex (remove distortion errors caused by lighting)
                    if cv2.isContourConvex(approx):
                        doc_cnt = approx
                        break 
            
            if doc_cnt is not None:
                break

        # =========================================================
        # MOTION TRACKING LOGIC (AUTO-CAPTURE)
        # =========================================================
        box_color = (0, 0, 255) # Default to Red border (Moving)

        if doc_cnt is not None:
            current_corners = self.order_points(doc_cnt.reshape(4, 2))
            
            if self.prev_corners is not None:
                # Calculate Euclidean distance between old and new corner coordinates
                dist = np.max(np.linalg.norm(current_corners - self.prev_corners, axis=1))
                if dist < self.STABLE_THRESHOLD:
                    self.stable_counter += 1
                else:
                    self.stable_counter = 0 
            else:
                self.stable_counter = 0
                
            self.prev_corners = current_corners
            
            # If the hand is steady enough for 3 frames -> Change border to Green and set flag
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
            # Draw the actual contour (Yellow) and draw 4 corner points (Red/Green)
            cv2.drawContours(orig, [c], -1, (0, 255, 255), 2)  
            cv2.drawContours(orig, [doc_cnt], -1, box_color, 3) 
            for point in doc_cnt:
                cv2.circle(orig, tuple(point[0]), 8, box_color, -1)
                
            process_time_ms = (time.time() - start_time) * 1000
            return orig, results, process_time_ms

        # =========================================================
        # STAGE 3: LINEAR ALGEBRA (HOMOGRAPHY)
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
        # STAGE 4: BRIGHTEN & INCREASE SHARPNESS (SHARPENING)
        # =========================================================
        
        # 1. Brighten and increase contrast (As you did)
        alpha = 1.2  
        beta = 25     
        scanned_bgr = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

        # 2. Sharpening Filter (Sharpening Kernel)
        # Create a 3x3 matrix to amplify color differences at the edge regions
        sharpen_kernel = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ])
        
        # Apply Convolution of the filter matrix to the image
        scanned_bgr = cv2.filter2D(scanned_bgr, -1, sharpen_kernel)

        process_time_ms = (time.time() - start_time) * 1000
        
        return scanned_bgr, results, process_time_ms