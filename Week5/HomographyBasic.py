import cv2
import numpy as np

# Load input image
img = cv2.imread('blackboard.jpg')

# List to store points clicked by the user
srcPoints = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(srcPoints) < 4:
        srcPoints.append([x, y])
        cv2.circle(img_display, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 corners", img_display)



def compute_homography(src, dst):
    """ Computes the homography matrix using the Direct Linear Transform (DLT) algorithm.

        Args:
            src: (4x2) array of (x, y) source points
            dst: (4x2) array of (x', y') destination points

        Returns:
            H: (3x3) homography matrix
    """
    n = src.shape[0]
    if n != 4:
        raise ValueError('This function only supports exactly 4 points for a unique solution.')

    A = []
    for i in range(n):
        x, y = src[i][0], src[i][1]
        x_p, y_p = dst[i][0], dst[i][1]
        A.append([-x, -y, -1,  0,  0,  0, x*x_p, y*x_p, x_p])
        A.append([ 0,  0,  0, -x, -y, -1, x*y_p, y*y_p, y_p])

    A = np.array(A)
    # Compute the SVD
    U, S, Vt = np.linalg.svd(A)
    # Homography is the last column of V (or row of V transposed)
    h = Vt[-1, :]
    # Reshape to 3x3
    H = h.reshape(3, 3)
    # Normalize so that H[2,2] == 1 (optional, for scale invariance)
    H = H / H[-1, -1]
    return H


# Clone image for display
img_display = img.copy()
cv2.imshow("Select 4 corners", img_display)
cv2.setMouseCallback("Select 4 corners", click_event)

# Wait for the user to click 4 corners
print("Please click 4 points clockwise or counter-clockwise on the image...")
while len(srcPoints) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

# Convert to NumPy array
srcPoints = np.array(srcPoints, dtype="float32")

# Define destination points for the homography (rectangle of desired size)
width = 400
height = 300
dstPoints = np.array([
    [0, 0],
    [width - 1, 0],
    [width - 1, height - 1],
    [0, height - 1]
], dtype="float32")

# Find the homography matrix
#H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC)
H = compute_homography(srcPoints, dstPoints)


print("Homography Matrix (H):\n", H)

# Warp the image using the estimated homography
warped_img = cv2.warpPerspective(img, H, (width, height))

# Show the result
cv2.imshow('Warped Image', warped_img)
cv2.imwrite('warped_output.jpg', warped_img)
print("Warped image saved as warped_output.jpg")
cv2.waitKey(0)
cv2.destroyAllWindows()