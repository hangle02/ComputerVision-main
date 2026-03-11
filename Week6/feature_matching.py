import cv2
img1 = cv2.imread('img1.jpg', 0)
img2 = cv2.imread('img2.jpg', 0)
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

img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
cv2.imshow('Matches', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()