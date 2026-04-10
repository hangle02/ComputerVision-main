import cv2

fgbg = cv2.createBackgroundSubtractorMOG2()
cap = cv2.VideoCapture('input.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    fgmask = fgbg.apply(frame)
    cv2.imshow('MOG2 Motion', fgmask)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()
