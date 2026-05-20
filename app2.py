import os
import cv2
import time
import base64
import numpy as np
import threading
from flask import Flask, render_template, Response, request, jsonify

from process_cv import ImageProcessor

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
cv2.setNumThreads(1)

app = Flask(__name__)

class VideoCamera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.source = None

    def start(self, source):
        self.stop()
        self.source = source
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame_bgr(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def get_frame_jpeg(self, quality=80):
        frame = self.get_frame_bgr()
        if frame is None:
            return None
        ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return jpeg.tobytes() if ret else None

    def stop(self):
        self.running = False
        if self.cap is not None:
            try: self.cap.release()
            except: pass
            self.cap = None
        with self.lock:
            self.frame = None

cameras = {1: VideoCamera(), 2: VideoCamera()}
processors = {1: ImageProcessor(), 2: ImageProcessor()}

@app.route('/')
def index():
    return render_template('index.html')

def create_blank_jpeg():
    img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes() if ret else b''

def mjpeg_generator(cam_id):
    cam = cameras.get(cam_id)
    if cam is None: return
    boundary = b'--frame'
    while True:
        frame_bytes = cam.get_frame_jpeg(70)
        if frame_bytes:
            yield (boundary + b'\r\n' b'Content-Type: image/jpeg\r\n' b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            blank = create_blank_jpeg()
            yield (boundary + b'\r\n' b'Content-Type: image/jpeg\r\n' b'Content-Length: ' + str(len(blank)).encode() + b'\r\n\r\n' + blank + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(mjpeg_generator(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source', methods=['POST'])
def set_source():
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 0))
    source = data.get('source', '').strip()
    if cam_id not in cameras: return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    try:
        if source == '':
            cameras[cam_id].stop()
            return jsonify({'ok': True, 'msg': 'stopped'})
        cameras[cam_id].start(source)
        return jsonify({'ok': True, 'msg': 'started'})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 0))
    selected_step = data.get('step', 'all')

    if cam_id not in cameras: return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    frame = cameras[cam_id].get_frame_bgr()
    if frame is None: return jsonify({'ok': False, 'error': 'no frame yet'}), 400

    try:
        processor = processors[cam_id]
        
        # CHẾ ĐỘ XUẤT HẾT CÁC BƯỚC (FULL PIPELINE DIALECT)
        if selected_step == 'all':
            stages_images, stages_times, results = processor.process_all_stages(frame)
            
            ret_orig, jpg_orig = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            image_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg_orig.tobytes()).decode('utf-8')
            
            stages_uris = {}
            for k, img in stages_images.items():
                ret_s, jpg_s = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ret_s:
                    stages_uris[k] = 'data:image/jpeg;base64,' + base64.b64encode(jpg_s.tobytes()).decode('utf-8')
            
            rounded_times = {k: round(v, 2) for k, v in stages_times.items()}
            total_time = round(sum(stages_times.values()), 2)
            
            return jsonify({
                'ok': True,
                'image': image_uri,
                'processed': stages_uris['all'],  # Backup cho khung hiển thị đơn
                'stages': stages_uris,
                'stages_times': rounded_times,
                'process_time_ms': total_time,
                'results': results,
                'step': 'all'
            })
        
        # CHẾ ĐỘ XEM ĐƠN LẺ TỪNG BƯỚC
        else:
            processed, results, process_time_ms = processor.process_frame(frame, step=selected_step)
            ret1, jpg1 = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            ret2, jpg2 = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ret1 or not ret2: return jsonify({'ok': False, 'error': 'encode_failed'}), 500
            
            image_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg1.tobytes()).decode('utf-8')
            processed_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg2.tobytes()).decode('utf-8')
            
            return jsonify({
                'ok': True,
                'image': image_uri,
                'processed': processed_uri,
                'process_time_ms': round(process_time_ms, 2),
                'results': results,
                'step': selected_step
            })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Processing failed: {str(e)}'}), 500

@app.route('/auto_capture', methods=['POST'])
def auto_capture():
    data = request.get_json() or {}
    cam_id = int(data.get('cam_id', 0))
    selected_step = data.get('step', 'all')
    if cam_id not in cameras: return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    frame = cameras[cam_id].get_frame_bgr()
    if frame is None: return jsonify({'ok': False, 'error': 'no frame yet'}), 400
    try:
        processor = processors[cam_id]
        processed, results, process_time_ms = processor.process_frame(frame, step=selected_step)
        ret, jpg = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ret: return jsonify({'ok': False, 'error': 'encode_failed'}), 500
        processed_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode('utf-8')
        return jsonify({'ok': True, 'captured': bool(results.get('is_stable', False)), 'processed': processed_uri, 'results': results, 'process_time_ms': round(process_time_ms, 2)})
    except Exception as e: return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/rotate', methods=['POST'])
def rotate():
    data = request.get_json() or {}
    image_b64 = data.get('image')
    if not image_b64: return jsonify({'ok': False, 'error': 'No image provided'}), 400
    try:
        encoded_data = image_b64.split(',')[1] if ',' in image_b64 else image_b64
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ret, buffer = cv2.imencode('.jpg', rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret: return jsonify({'ok': False, 'error': 'encode_failed'}), 500
        new_data_uri = 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')
        return jsonify({'ok': True, 'image': new_data_uri})
    except Exception as e: return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/status/<int:cam_id>')
def status(cam_id):
    if cam_id not in cameras: return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    frame = cameras[cam_id].get_frame_bgr()
    return jsonify({'ok': True, 'has_frame': frame is not None})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5006, threaded=True, use_reloader=False)