// Hàm kết nối IP Camera
function setSource(camId) {
    const sourceUrl = document.getElementById(`src-${camId}`).value;
    
    // Gửi link IP Camera xuống cho Flask
    fetch('/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, source: sourceUrl })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            // Thêm time stamp (?t=...) để ép trình duyệt tải video mới, không dùng cache
            document.getElementById(`video-${camId}`).src = `/video_feed/${camId}?t=${new Date().getTime()}`;
        } else {
            alert('Error connecting to camera: ' + data.error);
        }
    })
    .catch(err => console.error('Connection error:', err));
}

// Hàm dừng Camera
function stopSource(camId) {
    fetch('/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, source: '' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            // Xóa src để dừng hiển thị video
            document.getElementById(`video-${camId}`).removeAttribute('src');
        }
    })
    .catch(err => console.error(err));
}

// Hàm chụp và xử lý ảnh (Computer Vision + Linear Algebra)
function captureAndProcess(camId, stepName) {
    console.log(`Capturing from cam ${camId} with step: ${stepName}`);
    
    // Cập nhật nhãn (label) trên giao diện
    document.getElementById(`current-step-label`).innerText = stepName.toUpperCase();

    fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, step: stepName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            // Hiển thị ảnh gốc và ảnh đã qua thuật toán
            document.getElementById(`captured-${camId}`).src = data.image;
            document.getElementById(`fragment-${camId}`).src = data.processed;
            
            // Hiển thị thời gian chạy
            document.getElementById(`proc-time-${camId}`).innerText = `Process time: ${data.process_time_ms} ms`;
            
            // --- Hiển thị Ma trận Homography (nếu có) ---
            const matrixBox = document.getElementById(`matrix-display-${camId}`);
            if (data.results && data.results.homography_matrix) {
                let matrixStr = "Homography Matrix H (3x3):\n\n";
                data.results.homography_matrix.forEach(row => {
                    // Format các con số cho thẳng hàng
                    matrixStr += "[ " + row.map(val => val.toFixed(4).padStart(10)).join(", ") + " ]\n";
                });
                matrixBox.innerText = matrixStr;
                matrixBox.style.display = 'block'; // Hiện box lên
            } else {
                matrixBox.style.display = 'none';  // Ẩn box đi nếu bước này không có tính toán ma trận
            }
            
        } else {
            alert('Error processing image: ' + data.error);
            console.error(data.error);
        }
    })
    .catch(err => {
        alert('Network error occurred. Make sure the server is running.');
        console.error(err);
    });
}