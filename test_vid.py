import cv2
import time
from process_cv import ImageProcessor 

def test_tracking_only(video_path):
    processor = ImageProcessor()
    cap = cv2.VideoCapture(r'C:\ComputerVision-main\input2.mp4')
    
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở video: {video_path}")
        return

    print("🎬 Đang chạy chế độ: CHỈ HIỂN THỊ VIỀN TRACKING (CONTOURS)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # ĐIỂM MẤU CHỐT: Chỉ gọi bước 'contours'
        # Thuật toán sẽ tính toán xong và chỉ vẽ cái khung lên ảnh gốc rồi trả về ngay
        tracked_frame, results, _ = processor.process_frame(frame, step='contours')
        
        process_time_ms = (time.time() - start_time) * 1000
        fps = 1000.0 / process_time_ms if process_time_ms > 0 else 0

        # Vẽ bảng thông số siêu gọn nhẹ lên góc trái
        cv2.rectangle(tracked_frame, (5, 5), (250, 90), (0, 0, 0), -1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(tracked_frame, f"FPS: {fps:.1f}", (15, 35), font, 0.7, (0, 255, 0) if fps > 15 else (0, 0, 255), 2)
        
        stable = results.get("is_stable", False)
        cv2.putText(tracked_frame, f"Lock: {'ON' if stable else 'OFF'}", (15, 70), font, 0.7, (0, 255, 0) if stable else (0, 0, 255), 2)

        # Scale lại màn hình cho vừa mắt (chiều cao 720px)
        height, width = tracked_frame.shape[:2]
        if height > 720:
            ratio = 720 / float(height)
            tracked_frame = cv2.resize(tracked_frame, (int(width * ratio), 720))

        # Chỉ Show ĐÚNG 1 CỬA SỔ
        cv2.imshow("Document Tracking Live", tracked_frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Đổi tên file video của bạn vào đây
    VIDEO_FILE = "video_test_cua_ban.mp4" 
    test_tracking_only(VIDEO_FILE)