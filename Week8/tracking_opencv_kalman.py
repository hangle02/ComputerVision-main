import cv2
import numpy as np
from collections import deque

# Define lower and upper HSV bounds for the color to track (example: green)
lower_color = np.array([29, 86, 6])
upper_color = np.array([64, 255, 255])

# For showing trace paths
pts = deque(maxlen=64)
kalman_pts = deque(maxlen=64)

cap = cv2.VideoCapture(0)  # 0 = default webcam

# ------------ Kalman filter setup ------------
# 4 dynamic STATE parameters (x, y, dx, dy), 2 measured parameters (x, y)
kalman = cv2.KalmanFilter(4, 2)

# init the measurement matrix, H, only consider x and y
kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0]], np.float32)

#init transition matrix, A, x + vx, y + vy, vx and vy
kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                    [0, 1, 0, 1],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], np.float32)
#process noise covariance, Q
kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

#measurement noise covariance, R
kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

first_detected = False  # flag for Kalman initialization
# ---------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Blur to reduce noise, convert to HSV
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create mask based on color, then clean noise
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in mask
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        # Pick largest contour, compute min enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Kalman filter: first measurement, initialize state
            if not first_detected:
                kalman.statePre = np.array([[center[0]], [center[1]], [0.], [0.]], dtype=np.float32)
                kalman.statePost = kalman.statePre.copy()
                first_detected = True
            # Measurement update (correction)
            measured = np.array([[center[0]], [center[1]]], dtype=np.float32)
            kalman.correct(measured)


            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    # Predict filtered position (always, even if no detection)
    if first_detected:
        prediction = kalman.predict()
        kalman_center = (int(prediction[0].item()), int(prediction[1].item()))
        # Draw Kalman prediction (blue dot)
        cv2.circle(frame, kalman_center, 5, (255, 0, 0), -1)
        kalman_pts.appendleft(kalman_center)
    else:
        kalman_pts.appendleft(None)

    pts.appendleft(center)

    # Raw trace path (red)
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    # Kalman filtered trace path (blue)
    for i in range(1, len(kalman_pts)):
        if kalman_pts[i - 1] is None or kalman_pts[i] is None:
            continue
        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
        cv2.line(frame, kalman_pts[i - 1], kalman_pts[i], (255, 0, 0), thickness)

    cv2.imshow("Kalman Tracking", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()