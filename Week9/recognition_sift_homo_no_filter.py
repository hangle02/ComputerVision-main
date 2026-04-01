import cv2
import numpy as np

# --- SIFT setup ---
sift = cv2.SIFT_create()

# Load template image in grayscale
template = cv2.imread(r'Week9/template.png', cv2.IMREAD_GRAYSCALE)
if template is None:
    print("Cannot load template image!")
    exit()
kp_template, des_template = sift.detectAndCompute(template, None)
print("Template SIFT keypoints:", len(kp_template))

MIN_MATCH_COUNT = 4  # For homography, must be >= 4

cap = cv2.VideoCapture(0)
bf = cv2.BFMatcher()
object_id = 1  # Use a static ID for this scenario

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(frame_gray, None)

    detections = []
    match_feedback = ""
    if des_frame is not None and des_template is not None:
        matches = bf.knnMatch(des_template, des_frame, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        # Draw matches for debugging
        if len(good) > 0:
            draw_matches = cv2.drawMatches(template, kp_template, frame_gray, kp_frame, good, None, flags=2)
            cv2.imshow('Matches', draw_matches)

        if len(good) >= MIN_MATCH_COUNT:
            # Need at least 4 for homography!
            src_pts = np.float32([kp_template[m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1,1,2)
            M, mask2 = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            if M is not None:
                h, w = template.shape
                pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts, M)
                bbox = cv2.boundingRect(dst)
                x, y, b_w, b_h = bbox
                centroid = (int(x + b_w/2), int(y + b_h/2))
                detections.append((centroid, bbox))
                frame = cv2.polylines(frame, [np.int32(dst)], True, (0,255,0), 3, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + b_w, y + b_h), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {object_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                match_feedback = f"Object detected (good matches: {len(good)})"
            else:
                match_feedback = "Not enough matches (no homography found)"
        else:
            match_feedback = f"No SIFT match (good matches: {len(good)})"
    else:
        match_feedback = "No descriptors in frame or template"

    # Feedback text for debugging
    cv2.putText(
        frame, match_feedback, (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
    )

    cv2.imshow("SIFT Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()