import os
from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import base64
import numpy as np

# --- Import class xử lý ảnh từ file process.py ---
from process import ImageProcessor 
from camera import VideoCamera

# --- CRITICAL FIX: Ép luồng RTSP/HTTP dùng TCP ---
# Giúp tránh lỗi "method SETUP failed: 500" khi OpenCV cố kết nối qua UDP trước.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

app = Flask(__name__)

# Khởi tạo danh sách camera (hỗ trợ nhiều luồng nếu cần)
cameras = {
    1: VideoCamera(),
    2: VideoCamera()
}

# --- CÁC ĐƯỜNG DẪN (ROUTES) ---
@app.route('/')
def index():
    return render_template('index.html')

def create_blank_jpeg():
    """Tạo một khung hình xám dự phòng khi camera chưa sẵn sàng"""
    img = 128 * np.ones((240, 320, 3), dtype=np.uint8)
    ret, jpeg = cv2.imencode('.jpg', img)
    return jpeg.tobytes() if ret else b''

def mjpeg_generator(cam_id):
    """Hàm tạo luồng stream MJPEG liên tục xuống Web"""
    cam = cameras.get(cam_id)
    if cam is None:
        return
    boundary = b'--frame'
    while True:
        frame_bytes = cam.get_frame_jpeg()
        if frame_bytes:
            yield b'%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (boundary, len(frame_bytes), frame_bytes)
        else:
            blank = create_blank_jpeg()
            yield b'%s\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n%s\r\n' % (boundary, len(blank), blank)
        time.sleep(0.04)  # Giới hạn ~25 FPS để giảm tải CPU

@app.route('/video_feed/<int:cam_id>')
def video_feed(cam_id):
    return Response(mjpeg_generator(cam_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_source', methods=['POST'])
def set_source():
    """Nhận địa chỉ IP Camera từ giao diện web và khởi động luồng đọc"""
    data = request.get_json()
    cam_id = int(data.get('cam_id'))
    source = data.get('source', '').strip()
    
    if cam_id not in cameras:
        return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    
    if source == '':
        cameras[cam_id].stop()
        return jsonify({'ok': True, 'msg': 'stopped'})
    
    try:
        cameras[cam_id].start(source)
        return jsonify({'ok': True})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

@app.route('/capture', methods=['POST'])
def capture():
    """
    Chụp ảnh từ camera và đưa qua model xử lý
    Payload yêu cầu: { cam_id: int, step: str (tùy chọn) }
    """
    data = request.get_json()
    cam_id = int(data.get('cam_id'))
    
    # Lấy tham số step, mặc định là 'all'
    selected_step = data.get('step', 'all')
    
    if cam_id not in cameras:
        return jsonify({'ok': False, 'error': 'invalid cam_id'}), 400
    
    cam = cameras[cam_id]
    frame = cam.get_frame_bgr()
    if frame is None:
        return jsonify({'ok': False, 'error': 'no frame yet'}), 400

    # 1. Mã hóa ảnh Gốc sang Base64 để hiển thị
    ret, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ret:
        return jsonify({'ok': False, 'error': 'encode_failed'}), 500
    data_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg.tobytes()).decode('utf-8')

    # 2. Đưa ảnh qua ImageProcessor để chạy thuật toán Computer Vision & Linear Algebra
    try:
        processor = ImageProcessor()
        # Truyền tham số step để lấy đúng giai đoạn hình ảnh mong muốn
        processed, results, process_time_ms = processor.process_frame(frame, step=selected_step)
        
        # 3. Mã hóa ảnh Kết Quả sang Base64
        ret2, jpg2 = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret2:
            return jsonify({'ok': False, 'error': 'processed_encode_failed'}), 500
        processed_uri = 'data:image/jpeg;base64,' + base64.b64encode(jpg2.tobytes()).decode('utf-8')
        
        # 4. Trả toàn bộ dữ liệu về cho Frontend
        return jsonify({
            'ok': True, 
            'image': data_uri, 
            'processed': processed_uri, 
            'process_time_ms': round(process_time_ms, 2),
            'results': results, 
            'step': selected_step
        })
    except Exception as e:
        return jsonify({'ok': False, 'error': f'Processing failed: {str(e)}'}), 500
@app.route('/rotate', methods=['POST'])
def rotate():
    """
    Nhận ảnh base64 từ frontend, xoay 90 độ cùng chiều kim đồng hồ và trả về.
    """
    data = request.get_json()
    image_b64 = data.get('image')

    if not image_b64:
        return jsonify({'ok': False, 'error': 'No image provided'}), 400

    try:
        # Tách phần header 'data:image/jpeg;base64,' ra khỏi chuỗi
        encoded_data = image_b64.split(',')[1]
        
        # Decode base64 thành ma trận ảnh OpenCV
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Thực hiện xoay 90 độ cùng chiều kim đồng hồ
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Encode ngược lại thành base64
        ret, buffer = cv2.imencode('.jpg', rotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        new_b64 = base64.b64encode(buffer).decode('utf-8')
        new_data_uri = 'data:image/jpeg;base64,' + new_b64

        return jsonify({'ok': True, 'image': new_data_uri})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # Chạy Server Flask ở port 5006
    app.run(host='0.0.0.0', port=5006, threaded=True)