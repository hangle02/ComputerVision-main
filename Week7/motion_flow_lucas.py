import cv2
import numpy as np

# Load images (make sure they are the same size!)
img1 = cv2.imread('motion1.jpg')
img2 = cv2.imread('motion2.jpg')
if img1 is None or img2 is None:
    print("Cannot load images.")
    exit(1)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Create a grid of points (dense enough for visualization)
step = 10  # you can reduce the step to get an even denser grid
h, w = gray1.shape
y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1)
points = np.vstack((x, y)).T.astype(np.float32)

# Calculate Lucas-Kanade optical flow for these points
next_points, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points, None)

# Create an output image to draw flow
output = img2.copy()

# Draw the flow vectors
for (x1, y1), (x2, y2), good in zip(points, next_points, status.ravel()):
    if good:
        cv2.arrowedLine(output, (int(x1), int(y1)), (int(x2), int(y2)),
                        color=(0, 255, 0), thickness=1, tipLength=0.4)

cv2.imshow('Lucas-Kanade Dense-like Flow', output)
cv2.waitKey(0)
cv2.destroyAllWindows()