import cv2
import numpy as np
import os

os.makedirs('test_hsv_variations', exist_ok=True)

img = cv2.imread(r'C:\ComputerVision-main\Week11\leaf.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

hue_vals = [140, 160, 180]
sat_vals = [0, 20, 40, 60, 80, 100]
val_vals = [0, 20, 40, 60, 80, 100]

cnt = 0
for hs in hue_vals:
    for ss in sat_vals:
        for vs in val_vals:
            h_new = np.clip(h + hs, 0, 179)
            s_new = np.clip(s * (1 - ss / 100), 0, 255)
            v_new = np.clip(v * (vs / 100 + 0.5), 0, 255)

            hsv_new = cv2.merge([h_new.astype(np.uint8), s_new.astype(np.uint8), v_new.astype(np.uint8)])
            result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

            filename = f'output_h{hs}_s{ss}_v{vs}.jpg'
            cv2.imwrite(filename, result)
            cnt += 1
            print(f'{cnt}. {filename}')

print(f'Tong: {cnt} anh')