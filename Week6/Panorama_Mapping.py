import cv2
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