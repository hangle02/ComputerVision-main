import cv2
import numpy as np

# 1. Read image and convert to grayscale
img = cv2.imread('img2.jpg') # Replace with your image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cornerHarris requires the input to be float32
gray = np.float32(gray)

# 2. Apply Harris Corner Detection
# Parameters: (img, blockSize, ksize, k)
# - blockSize: Size of the neighborhood considered for corner detection (usually 2)
# - ksize: Aperture parameter of the Sobel derivative used (usually 3)
# - k: Harris detector free parameter in the equation (usually 0.04 to 0.06)
dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 3. Dilate the result to make the corners easier to see on the image
dst = cv2.dilate(dst, None)

# 4. Thresholding: Mark the corners on the original image
# We only keep responses that are greater than 1% (0.01) of the maximum response
threshold = 0.01 * dst.max()
img[dst > threshold] = [0, 0, 255] # Mark corners in Red (BGR)

img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

cv2.imshow('OpenCV Harris Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()