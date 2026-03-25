import cv2
import numpy as np
from scipy.spatial import distance

def find_centroids(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Use inverted threshold for black objects on white background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 50:
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
    return centroids

def simple_nearest_neighbor_assignment(centroids1, centroids2):
    associations = []
    centroids2_copy = centroids2.copy()
    for c1 in centroids1:
        # Compute distance to all centroids in frame2
        if not centroids2_copy:
            break
        dists = [np.linalg.norm(np.array(c1) - np.array(c2)) for c2 in centroids2_copy]
        min_idx = int(np.argmin(dists))
        c2 = centroids2_copy[min_idx]
        associations.append((c1, c2))
        del centroids2_copy[min_idx]  # Remove to prevent multiple assignment
    return associations

# Load your two frames
img1 = cv2.imread('motion1.jpg')
img2 = cv2.imread('motion2.jpg')
img2_assoc = img2.copy()

# 1. Detect centroids in both frames
centroids1 = find_centroids(img1)
centroids2 = find_centroids(img2)

print("Frame 1 centroids:", centroids1)
print("Frame 2 centroids:", centroids2)

# 2. Associate objects using nearest neighbor distance
associations = simple_nearest_neighbor_assignment(centroids1, centroids2)

# 3. Visualize: draw all centroids and association lines on img2
for (cx, cy) in centroids1:
    cv2.circle(img2_assoc, (cx, cy), 7, (0, 255, 0), 2)  # Green: previous positions

for (cx, cy) in centroids2:
    cv2.circle(img2_assoc, (cx, cy), 7, (0, 0, 255), 2)  # Red: current positions

for (c1, c2) in associations:
    cv2.line(img2_assoc, c1, c2, (255, 0, 0), 2)  # Blue lines = associations

# Show the images
cv2.imshow('Frame 1', img1)
cv2.imshow('Frame 2 with Associations', img2_assoc)
cv2.waitKey(0)
cv2.destroyAllWindows()