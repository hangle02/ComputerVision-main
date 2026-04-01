import cv2
import numpy as np
from collections import OrderedDict

# -- Tuning parameters --
lower_color = np.array([29, 86, 6])   # Green lower HSV
upper_color = np.array([64, 255, 255]) # Green upper HSV
min_area = 200  # Minimum contour area to be considered an object
max_distance = 50  # Maximum pixel distance for maintaining tracking

# -- Tracker structures --
class TrackedObject:
    def __init__(self, object_id, centroid, bbox):
        self.id = object_id
        self.bbox = bbox
        self.centroid = centroid
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.first_detected = True
        self.kalman.statePre = np.array([[centroid[0]], [centroid[1]], [0.0], [0.0]], dtype=np.float32)
        self.kalman.statePost = self.kalman.statePre.copy()
        self.missing = 0  # frames not detected

    def update(self, centroid, bbox):
        measured = np.array([[centroid[0]], [centroid[1]]], dtype=np.float32)
        self.kalman.correct(measured)
        self.first_detected = False
        self.centroid = centroid
        self.bbox = bbox
        self.missing = 0

    def predict(self):
        pred = self.kalman.predict()
        return (int(pred[0][0]), int(pred[1][0]))

# -- Multi-object tracker --
class MultiTracker:
    def __init__(self):
        self.objects = OrderedDict()
        self.next_object_id = 1

    def update(self, detections):
        assigned = set()
        # If no current tracked objects, add all detections as new tracked objects
        if len(self.objects) == 0:
            for centroid, bbox in detections:
                self.objects[self.next_object_id] = TrackedObject(self.next_object_id, centroid, bbox)
                self.next_object_id += 1
            return

        # Prepare cost matrix for matching detected centroids to existing tracked objects
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].centroid for oid in object_ids]
        object_matched = [False] * len(object_ids)
        detection_matched = [False] * len(detections)

        if object_centroids and detections:
            distances = np.zeros((len(object_centroids), len(detections)), dtype=np.float32)
            for i, obj_c in enumerate(object_centroids):
                for j, (det_c, _) in enumerate(detections):
                    distances[i, j] = np.linalg.norm(np.array(obj_c) - np.array(det_c))

            # Greedy assignment of detections to objects based on minimum distance
            while True:
                min_val = np.min(distances)
                if min_val > max_distance:
                    break
                idx = np.argmin(distances)
                i, j = np.unravel_index(idx, distances.shape)
                if object_matched[i] or detection_matched[j]:
                    distances[i, j] = np.inf
                    continue
                # Update this object with this detection
                obj = self.objects[object_ids[i]]
                obj.update(detections[j][0], detections[j][1])
                object_matched[i] = True
                detection_matched[j] = True
                distances[i, :] = np.inf
                distances[:, j] = np.inf

            # Unmatched objects: mark as missing
            for i, matched in enumerate(object_matched):
                if not matched:
                    self.objects[object_ids[i]].missing += 1
            # Unmatched detections: create new tracked objects
            for j, matched in enumerate(detection_matched):
                if not matched:
                    centroid, bbox = detections[j]
                    self.objects[self.next_object_id] = TrackedObject(self.next_object_id, centroid, bbox)
                    self.next_object_id += 1
        else:
            # If nothing to match, mark all tracked objects as missing (increase their missing frame count)
            for obj in self.objects.values():
                obj.missing += 1
            # Add new detections as new objects
            for centroid, bbox in detections:
                self.objects[self.next_object_id] = TrackedObject(self.next_object_id, centroid, bbox)
                self.next_object_id += 1

        # Remove objects missing for too many frames
        for object_id in list(self.objects.keys()):
            if self.objects[object_id].missing > 20:
                del self.objects[object_id]

# -- MAIN LOOP --
cap = cv2.VideoCapture(0)
tracker = MultiTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            detections.append((center, (x, y, w, h)))

    # Update tracker with all detected objects (centroid + bbox)
    tracker.update(detections)

    # Draw tracked objects and their Kalman prediction
    for obj in tracker.objects.values():
        if obj.missing > 0:
            # Not detected currently, just show prediction
            kalman_center = obj.predict()
            cv2.circle(frame, kalman_center, 5, (255, 0, 0), -1)
        else:
            # Detected, display tracked info
            x, y, w, h = obj.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.circle(frame, obj.centroid, 5, (0, 0, 255), -1)
            kalman_center = obj.predict()
            cv2.circle(frame, kalman_center, 5, (255, 0, 0), -1)
            cv2.putText(frame, f"ID: {obj.id}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Multi-Object Kalman Tracking + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()