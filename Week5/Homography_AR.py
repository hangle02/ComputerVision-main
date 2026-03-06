import cv2
import numpy as np

# Load images
big_image = cv2.imread(r'C:\ComputerVision-main\Week5\blackboard.jpg')
small_image = cv2.imread(r"C:\ComputerVision-main\Week5\small.jpg")
if big_image is None or small_image is None:
    print("Error loading images. Ensure 'big.jpg' and 'small.jpg' exist.")
    exit()

# User clicks 4 points on the big image
points = []

def click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        cv2.circle(big_show, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4 corners", big_show)

big_show = big_image.copy()
cv2.imshow("Select 4 corners", big_show)
cv2.setMouseCallback("Select 4 corners", click_point)
print("Click 4 destination points in the big image (where corners of small image will go).")
while len(points) < 4:
    cv2.waitKey(1)
cv2.destroyAllWindows()

dst_points = np.array(points, dtype=np.float32)

h_small, w_small = small_image.shape[:2]
src_points = np.array([[0, 0], [w_small-1, 0], [w_small-1, h_small-1], [0, h_small-1]], dtype=np.float32)

# Compute homography from small to big
H, _ = cv2.findHomography(src_points, dst_points)

# Warp the small image to the selected quadrilateral
warped_small = cv2.warpPerspective(small_image, H, (big_image.shape[1], big_image.shape[0]))

# Create a mask and inverse mask for blending
mask = np.zeros((big_image.shape[0], big_image.shape[1]), dtype=np.uint8)
cv2.fillConvexPoly(mask, dst_points.astype(int), 255)
mask_inv = cv2.bitwise_not(mask)

# Black-out region on big image
big_bg = cv2.bitwise_and(big_image, big_image, mask=mask_inv)
# Isolate the warped area from small image
small_fg = cv2.bitwise_and(warped_small, warped_small, mask=mask)
# Combine
composite = cv2.add(big_bg, small_fg)

cv2.imshow("Composite Result", composite)
cv2.imwrite("image_with_small_inserted.jpg", composite)
print("Saved as 'image_with_small_inserted.jpg'")
cv2.waitKey(0)
cv2.destroyAllWindows()