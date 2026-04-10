import cv2
import numpy as np

# ---- Minimal SORT tracker ---- #
from collections import deque
class Track:
    def __init__(self, id, bbox):
        self.id = id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.hits = 1
        self.no_losses = 0
        self.trace = deque(maxlen=20)
        self.trace.append(bbox)

class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_no_losses=3):
        self.next_id = 0
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_no_losses = max_no_losses

    def iou(self, bb_test, bb_gt):
        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
                 + (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return o

    def update(self, detections):
        assigned_detections = []
        # Associate detections with existing tracks
        for track in self.tracks:
            best_iou = 0
            best_det = None
            for i, det in enumerate(detections):
                iou_val = self.iou(det, track.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_det = i
            if best_iou > self.iou_threshold and best_det is not None:
                track.bbox = detections[best_det]
                track.hits += 1
                track.no_losses = 0
                track.trace.append(detections[best_det])
                assigned_detections.append(best_det)
            else:
                track.no_losses += 1

        # Remove lost tracks
        self.tracks = [t for t in self.tracks if t.no_losses <= self.max_no_losses]

        # Add new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i not in assigned_detections:
                self.tracks.append(Track(self.next_id, det))
                self.next_id += 1

        return self.tracks

# ---- Load MobileNet-SSD ---- #
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                               'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

cap = cv2.VideoCapture(0)  # or 'input.mp4'

tracker = SimpleTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect People
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                 (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        if confidence > 0.5 and CLASSES[idx] == "person":
            box = detections[0, 0, i, 3:7] * np.array([
                frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            boxes.append(box.astype("int"))

    # Update Tracker
    tracks = tracker.update(boxes)

    # Draw tracked persons and IDs
    for track in tracks:
        x1, y1, x2, y2 = track.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track.id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow('Person Detection & Tracking', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()