import cv2
import numpy as np

# Download the model files from: https://github.com/chuanqi305/MobileNet-SSD
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                               'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)  # or 'input.mp4'

while True:
    ret, frame = cap.read()
    if not ret: break
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, 
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if confidence > 0.5 and CLASSES[idx] == "person":
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
    cv2.imshow('MobileNetSSD Person Detection', frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()