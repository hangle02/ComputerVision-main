import cv2
import numpy as np

# Load images
img_left = cv2.imread('left.jpg')
img_right = cv2.imread('right.jpg')
if img_left is None or img_right is None:
    print("Failed to load images.")
    exit()

# 1. Detect ORB keypoints and descriptors.
orb = cv2.ORB_create(2000)
kp1, des1 = orb.detectAndCompute(img_left, None)
kp2, des2 = orb.detectAndCompute(img_right, None)

# 2. Match descriptors using Bruteforce and Lowe's ratio test.
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)  # find the top two matches for each descriptor

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    print("Not enough good matches!")
    exit()

# 3. Build point correspondences for Homography
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

# 4. Calculate Homography using RANSAC.
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
if H is None:
    print("Homography could not be computed!")
    exit()

print("Homography Matrix:\n", H)

# 5. Warp left image into right image's plane
height_left, width_left = img_left.shape[:2]
height_right, width_right = img_right.shape[:2]

# We want a panorama wide enough to hold both
result_width = width_left + width_right
result_height = max(height_left, height_right)

# Warp the left image
warped_left = cv2.warpPerspective(img_left, H, (result_width, result_height))

# Place the right image on top of the warped panorama
result = warped_left.copy()
result[0:height_right, 0:width_right] = img_right

# Optional: Crop out unused/black areas
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])
panorama = result[y:y+h, x:x+w]

# Show and save the results
cv2.imshow("Stitched Panorama", panorama)
cv2.imwrite("stitched_panorama.jpg", panorama)
print("Stitched panorama saved as stitched_panorama.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()