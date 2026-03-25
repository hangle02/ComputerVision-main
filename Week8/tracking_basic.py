import cv2
import numpy as np

def find_largest_centroid(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Invert threshold: now black (object) becomes white in binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

# Read your images
img1 = cv2.imread('motion1.jpg')  # or .jpg depending on your filename
img2 = cv2.imread('motion2.jpg')

centroid1 = find_largest_centroid(img1)
centroid2 = find_largest_centroid(img2)

img2_result = img2.copy()
if centroid1:
    cv2.circle(img2_result, centroid1, 5, (0, 255, 0), -1)  # Green: previous
if centroid2:
    cv2.circle(img2_result, centroid2, 5, (0, 0, 255), -1)  # Red: current
if centroid1 and centroid2:
    cv2.line(img2_result, centroid1, centroid2, (255, 0, 0), 2)  # Blue tracking line

cv2.imshow('Frame 1', img1)
cv2.imshow('Frame 2 with tracking', img2_result)
cv2.waitKey(0)
cv2.destroyAllWindows()