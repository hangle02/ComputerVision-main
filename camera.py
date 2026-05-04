import cv2
import threading
import time


class VideoCamera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.source = None

    def start(self, source):
        self.stop()
        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.02)

    def get_frame_bgr(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_jpeg(self):
        frame = self.get_frame_bgr()
        if frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes() if ret else None

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame = None