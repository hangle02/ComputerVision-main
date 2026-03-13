import cv2 as cv
import numpy as np

img = cv.imread('test.jpg', cv.IMREAD_GRAYSCALE)

siflt = cv.SIFT_create()
keypoints, descriptors = siflt.detectAndCompute(img, None)

img_with_keypoints = cv.drawKeypoints(img, 
                                      keypoints, 
                                      None, 
                                      flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
print(descriptors(0))
cv.imshow('SIFT Keypoints', img_with_keypoints)
cv.waitKey(0)
cv.destroyAllWindows()