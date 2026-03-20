import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def lucas_kanade_optical_flow(img1, img2, window_size=5):
    #Convert to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    #Gradient x,y and t
    Ix = 0.5 * (np.gradient(img1, axis=1) + np.gradient(img2, axis=1))
    Iy = 0.5 * (np.gradient(img1, axis=0) + np.gradient(img2, axis=0))
    It = img2 - img1

    half_w = window_size // 2
    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    h, w = img1.shape
    for y in range(half_w, h - half_w):
        for x in range(half_w, w - half_w):
            # Window (x,y)
            Ix_win = Ix[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            Iy_win = Iy[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()
            It_win = It[y-half_w:y+half_w+1, x-half_w:x+half_w+1].flatten()

            A = np.stack((Ix_win, Iy_win), axis=1)
            b = -It_win

            ATA = A.T @ A   # Matrix 2x2
            ATb = A.T @ b   # Vector 2x1

            # Check if A is valid matrix
            if np.linalg.det(ATA) > 1e-4:
                nu = np.linalg.inv(ATA) @ ATb
                u[y, x] = nu[0]
                v[y, x] = nu[1]
            else:
                u[y, x] = 0
                v[y, x] = 0

    return u, v

#read inputs
img1 = imageio.imread('motion1.jpg')
img2 = imageio.imread('motion2.jpg')
if img1.ndim == 3:
    img1 = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
if img2.ndim == 3:
    img2 = np.dot(img2[...,:3], [0.299, 0.587, 0.114])

u, v = lucas_kanade_optical_flow(img1, img2, window_size=7)



plt.figure(figsize=(8,8))
plt.imshow(img2, cmap='gray')

step = 10
arrow_scale = 8  # <-- số lớn để mũi tên dài rõ ràng
Y, X = np.mgrid[step//2:img1.shape[0]:step, step//2:img1.shape[1]:step]
for x0, y0 in zip(X.flatten(), Y.flatten()):
    dx = u[y0, x0]*arrow_scale
    dy = v[y0, x0]*arrow_scale
    plt.arrow(x0, y0, dx, dy, color='red', width=1.0, head_width=4, head_length=6, length_includes_head=True)

plt.title('Lucas-Kanade Optical Flow\n(Arrow from motion1 position to predicted motion2 position)')
plt.show()