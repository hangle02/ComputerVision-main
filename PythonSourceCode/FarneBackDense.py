import cv2

cap = cv2.VideoCapture('input.mp4')
_, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    motion_mask = (mag > 1).astype('uint8')*255
    cv2.imshow('Motion Mask', motion_mask)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    prev_gray = gray
cap.release()
cv2.destroyAllWindows()