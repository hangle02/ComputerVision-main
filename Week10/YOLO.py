from ultralytics import YOLO
import cv2

# Load model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.model.names[cls]
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
    cv2.imshow("YOLOv8", frame)
    if cv2.waitKey(1) & 0xFF == 27: break
cap.release()
cv2.destroyAllWindows()