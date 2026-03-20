import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

def horn_schunck(img1, img2, alpha=1.0, num_iter=100):
    """Horn-Schunck algorithm, pure numpy, for two grayscale images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    fx = 0.25 * (np.roll(img1, -1, axis=1) - np.roll(img1, 1, axis=1) +
                 np.roll(img2, -1, axis=1) - np.roll(img2, 1, axis=1))
    fy = 0.25 * (np.roll(img1, -1, axis=0) - np.roll(img1, 1, axis=0) +
                 np.roll(img2, -1, axis=0) - np.roll(img2, 1, axis=0))
    ft = 0.25 * (img2 - img1 +
                 np.roll(img2, -1, axis=1) - np.roll(img1, -1, axis=1) +
                 np.roll(img2, -1, axis=0) - np.roll(img1, -1, axis=0))

    u = np.zeros_like(img1)
    v = np.zeros_like(img1)

    for it in range(num_iter):
        u_avg = (
            np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
            np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)
        ) / 4.0
        v_avg = (
            np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
            np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1)
        ) / 4.0

        der = fx * u_avg + fy * v_avg + ft
        denom = alpha**2 + fx**2 + fy**2
        u = u_avg - fx * der / denom
        v = v_avg - fy * der / denom

    return u, v

# Read and prepare images
img1 = imageio.imread('motion1.jpg')
img2 = imageio.imread('motion2.jpg')

if img1.ndim == 3:  # Convert to grayscale if needed
    img1 = np.dot(img1[...,:3], [0.299, 0.587, 0.114])
if img2.ndim == 3:
    img2 = np.dot(img2[...,:3], [0.299, 0.587, 0.114])

u, v = horn_schunck(img1, img2, alpha=1.0, num_iter=150)

flow_scale = 8  # Increase this to make arrows longer!

# ---- Choose the point to highlight (object location in motion1.png) ----
# Use the center of the image, or replace with your object's coordinate
object_y = img1.shape[0] // 2  # vertical index
object_x = img1.shape[1] // 2  # horizontal index

u_obj = u[object_y, object_x]
v_obj = v[object_y, object_x]

end_x = int(object_x + u_obj * flow_scale)
end_y = int(object_y + v_obj * flow_scale)

print(f"Object moves from ({object_x}, {object_y}) to ({end_x}, {end_y}) (scaled flow vector shown)")

# ---- Draw dense flow field, now with better visibility ----
plt.figure(figsize=(8,8))
plt.imshow(img2, cmap='gray')

step = 10
Y, X = np.mgrid[step//2:img1.shape[0]:step, step//2:img1.shape[1]:step]
plt.quiver(X, Y, u[Y,X]*flow_scale, v[Y,X]*flow_scale, color='lime', angles='xy', scale=1, scale_units='xy', width=0.004)

# ---- Draw the object's flow vector thick and red ----
plt.arrow(object_x, object_y, u_obj*flow_scale, v_obj*flow_scale,
          head_width=8, head_length=12, fc='red', ec='red', linewidth=3)

plt.scatter([object_x], [object_y], color='yellow', s=80, label='Object start')
plt.scatter([end_x], [end_y], color='red', s=80, label='Object end')
plt.legend(loc='upper right')
plt.title("Horn-Schunck optical flow\n(object arrow: yellow to red)")
plt.show()