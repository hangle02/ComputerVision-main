import cv2
import numpy as np

img1 = cv2.imread('right.jpg', 0)
img2 = cv2.imread('left.jpg', 0)
# Scale both images down by 50%
img1 = cv2.resize(img1, (0, 0), fx=0.5, fy=0.4)
img2 = cv2.resize(img2, (0, 0), fx=0.4, fy=0.4)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

if len(good) > 10:
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in good]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
else:
    print("Not enough matches found.")
	
	
height, width = img2.shape
result = cv2.warpPerspective(img1, H, (width * 2, height))
result[0:height, 0:width] = img2
cv2.imshow('Panorama', result)
cv2.waitKey(0)
cv2.destroyAllWindows()