import cv2
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Load image
img = cv2.imread(r'C:\\ComputerVision-main\\Week11\\leaf.jpg')  # Reads image in BGR format (OpenCV default)

# Step 2: Convert to HSV color space
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step 3: Split into H, S, V channels
h, s, v = cv2.split(img_hsv)

# Step 4: Display original and HSV channels
plt.figure(figsize=(12,4))
plt.subplot(1,4,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original')
plt.axis('off')
plt.subplot(1,4,2)
plt.imshow(h, cmap='hsv')
plt.title('Hue')
plt.axis('off')
plt.subplot(1,4,3)
plt.imshow(s, cmap='gray')
plt.title('Saturation')
plt.axis('off')
plt.subplot(1,4,4)
plt.imshow(v, cmap='gray')
plt.title('Value')
plt.axis('off')
plt.tight_layout()
plt.show()

# Step 5: change hsv an convert to RGB
hs = 140; ss = 0; vs = 0
h_new = np.clip(h + hs, 0, 179)
s_new = np.clip(s * (1 - ss / 100), 0, 255)
v_new = np.clip(v * (vs / 100 + 0.5), 0, 255)

hsv_new = cv2.merge([h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)])
result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

filename = f'output_h{hs}_s{ss}_v{vs}.jpg'
cv2.imwrite(filename, result)