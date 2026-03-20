import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load your first image
img1 = cv2.imread('motion1.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

# Synthetically create a shifted frame (by 5 pixels right)
M = np.float32([[1, 0, 5], [0, 1, 0]])
img2 = cv2.imread('motion2.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)

gray1 = img1
gray2 = img2

# Gradients
Ix = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
Iy = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
It = gray2 - gray1

#Plot
plt.figure(figsize=(12,4))

plt.subplot(1, 3, 1)
plt.title('Ix (Sobel X)')
plt.imshow(Ix, cmap='seismic', vmin=-np.max(np.abs(Ix)), vmax=np.max(np.abs(Ix)))
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 3, 2)
plt.title('Iy (Sobel Y)')
plt.imshow(Iy, cmap='seismic', vmin=-np.max(np.abs(Iy)), vmax=np.max(np.abs(Iy)))
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.subplot(1, 3, 3)
plt.title('It (Temporal Diff)')
plt.imshow(It, cmap='bwr', vmin=-np.max(np.abs(It)), vmax=np.max(np.abs(It)))
plt.axis('off')
plt.colorbar(fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

step = 5
win_size = 10
h, w = gray1.shape
output = cv2.cvtColor(gray2.astype(np.uint8), cv2.COLOR_GRAY2BGR)

count = 0

for y in range(step//2, h-step//2, step):
    for x in range(step//2, w-step//2, step):
        x0, y0 = x, y
        x1 = x0 - win_size//2
        x2 = x0 + win_size//2 + 1
        y1 = y0 - win_size//2
        y2 = y0 + win_size//2 + 1

        if x1 < 0 or x2 > w or y1 < 0 or y2 > h:
            continue

        Ix_win = Ix[y1:y2, x1:x2].flatten()
        Iy_win = Iy[y1:y2, x1:x2].flatten()
        It_win = It[y1:y2, x1:x2].flatten()

        A = np.vstack((Ix_win, Iy_win)).T
        b = -It_win

        ATA = A.T @ A
        if np.linalg.cond(ATA) < 1e4:
            nu = np.linalg.inv(ATA) @ (A.T @ b)
            dx, dy = nu[0], nu[1]
            if np.hypot(dx, dy) > 0.5:
                pt1 = (int(x0), int(y0))
                pt2 = (int(x0 + dx), int(y0 + dy))
                cv2.arrowedLine(output, pt1, pt2, color=(0, 255, 0), thickness=10, tipLength=0.4)
                count += 1


if count == 0:
    print("No flow detected! Try a larger shift or check your images.")

cv2.imshow('Windowed Optical Flow (Basic Lucas-Kanade)', output)
cv2.waitKey(0)
cv2.destroyAllWindows()