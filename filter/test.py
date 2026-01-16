# import cv2
# img = cv2.imread('input.bmp')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('output_grayscale2.bmp', gray)

# import cv2
# img = cv2.imread('input2.bmp')
# blur = cv2.GaussianBlur(img, (9, 9), 0)
# cv2.imwrite('output_gaussian.bmp', blur)

# import cv2
# img = cv2.imread('input2.bmp')
# median = cv2.medianBlur(img, 5)
# cv2.imwrite('output_median.bmp', median)

# import cv2
# img = cv2.imread('input2.bmp', 0)
# laplacian = cv2.Laplacian(img, cv2.CV_64F)
# laplacian = cv2.convertScaleAbs(laplacian)
# cv2.imwrite('output_laplacian.bmp', laplacian)

# import cv2
# import numpy as np
# img = cv2.imread('output_grayscale.bmp')
# kernel = np.array([[0, 1, 0],
#                     [1, 1, 1],
#                     [0, 1, 0]])
# sharpened = cv2.filter2D(img, -1, kernel)
# cv2.imwrite('output_sharpen.bmp', sharpened)

# import cv2
# img = cv2.imread('input2.bmp')
# bilateral = cv2.bilateralFilter(img, 9, 75, 75)
# cv2.imwrite('output_bilateral.bmp', bilateral)

import cv2
img = cv2.imread('input2.bmp', 0)
_, binary = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite('output_binary.bmp', binary)
