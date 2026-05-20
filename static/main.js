function setSource(camId) {
    const sourceUrl = document.getElementById(`src-${camId}`).value;
    fetch('/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, source: sourceUrl })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            document.getElementById(`video-${camId}`).src = `/video_feed/${camId}?t=${new Date().getTime()}`;
        } else {
            alert('Error connecting to camera: ' + data.error);
        }
    })
    .catch(err => console.error('Connection error:', err));
}

function stopSource(camId) {
    fetch('/set_source', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, source: '' })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            document.getElementById(`video-${camId}`).removeAttribute('src');
        }
    })
    .catch(err => console.error(err));
}

// HÀM CHỤP VÀ ĐỔ DỮ LIỆU ĐA TẦNG (MULTI-STAGE PLOTTING)
function captureAndProcess(camId, stepName) {
    console.log(`Capturing from cam ${camId} with step: ${stepName}`);
    document.getElementById(`current-step-label`).innerText = stepName.toUpperCase();

    fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, step: stepName })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            document.getElementById(`captured-${camId}`).src = data.image;
            const gridBox = document.getElementById(`all-steps-grid-${camId}`);
            
            if (stepName === 'all' && data.stages) {
                // 1. Mở khối Grid 2x2 lên công khai
                gridBox.style.display = 'block';
                
                // 2. Điền data ảnh base64 vào từng ô lưới
                document.getElementById(`step-edges-${camId}`).src = data.stages.edges;
                document.getElementById(`step-contours-${camId}`).src = data.stages.contours;
                document.getElementById(`step-warp-${camId}`).src = data.stages.warp;
                document.getElementById(`step-all-${camId}`).src = data.stages.all;
                
                // 3. Hiển thị mốc thời gian độc lập (individual steps execution time)
                document.getElementById(`time-edges-${camId}`).innerText = `Time: ${data.stages_times.edges} ms`;
                document.getElementById(`time-contours-${camId}`).innerText = `Time: ${data.stages_times.contours} ms`;
                document.getElementById(`time-warp-${camId}`).innerText = `Time: ${data.stages_times.warp} ms`;
                document.getElementById(`time-all-${camId}`).innerText = `Time: ${data.stages_times.all} ms`;
                
                // 4. Đồng bộ lên khung kết quả đơn phía trên
                document.getElementById(`fragment-${camId}`).src = data.stages.all;
                document.getElementById(`proc-time-${camId}`).innerText = `Total Pipeline Time: ${data.process_time_ms} ms`;
            } else {
                // Nếu chỉ click xem lẻ bước 1, 2 hoặc 3, ẩn grid 2x2 đi
                gridBox.style.display = 'none';
                document.getElementById(`fragment-${camId}`).src = data.processed;
                document.getElementById(`proc-time-${camId}`).innerText = `Process time: ${data.process_time_ms} ms`;
            }
            
            const matrixBox = document.getElementById(`matrix-display-${camId}`);
            if (data.results && data.results.homography_matrix) {
                let matrixStr = "Homography Matrix H (3x3):\n\n";
                data.results.homography_matrix.forEach(row => {
                    matrixStr += "[ " + row.map(val => val.toFixed(4).padStart(10)).join(", ") + " ]\n";
                });
                matrixBox.innerText = matrixStr;
                matrixBox.style.display = 'block';
            } else {
                matrixBox.style.display = 'none';
            }
        } else {
            alert('Error processing image: ' + data.error);
        }
    })
    .catch(err => {
        alert('Network error occurred. Make sure the server is running.');
        console.error(err);
    });
}

function rotateImage(camId) {
    const imgElement = document.getElementById(`fragment-${camId}`);
    const stepAllImg = document.getElementById(`step-all-${camId}`);
    const currentBase64 = imgElement.src;

    if (!currentBase64 || !currentBase64.startsWith('data:image')) {
        alert("Please capture an image first!");
        return;
    }

    const originalTimeText = document.getElementById(`proc-time-${camId}`).innerText;
    document.getElementById(`proc-time-${camId}`).innerText = "Rotating...";

    fetch('/rotate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: currentBase64 })
    })
    .then(response => response.json())
    .then(data => {
        if (data.ok) {
            imgElement.src = data.image;
            if (stepAllImg) stepAllImg.src = data.image; // Xoay cả ảnh trong Grid
            document.getElementById(`proc-time-${camId}`).innerText = "Rotated successfully!";
            setTimeout(() => {
                document.getElementById(`proc-time-${camId}`).innerText = originalTimeText;
            }, 2000);
        } else {
            alert('Error rotating image: ' + data.error);
        }
    })
    .catch(err => {
        console.error(err);
        alert("Network error while rotating.");
    });
}

let autoScanActive = false;

function toggleAutoScan(camId) {
    const btn = document.getElementById(`auto-btn-${camId}`);
    if (autoScanActive) {
        autoScanActive = false;
        btn.innerText = "Auto Scan: OFF";
        btn.style.backgroundColor = "#7f8c8d";
        document.getElementById(`current-step-label`).innerText = "Auto Scan Disabled";
    } else {
        autoScanActive = true;
        btn.innerText = "Auto Scan: ON (Detecting...)";
        btn.style.backgroundColor = "#e74c3c";
        triggerAutoScanLoop(camId);
    }
}

function triggerAutoScanLoop(camId) {
    if (!autoScanActive) return;

    fetch('/capture', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_id: camId, step: 'contours' })
    })
    .then(res => res.json())
    .then(data => {
        if (data.ok) {
            document.getElementById(`captured-${camId}`).src = data.image;
            document.getElementById(`fragment-${camId}`).src = data.processed;
            
            if (data.results && data.results.is_stable) {
                autoScanActive = false;
                const btn = document.getElementById(`auto-btn-${camId}`);
                btn.innerText = "Auto Scan: OFF";
                btn.style.backgroundColor = "#7f8c8d";
                
                // Tự động gọi bản Full Grid phá vỡ các bước
                captureAndProcess(camId, 'all');
            } else {
                setTimeout(() => triggerAutoScanLoop(camId), 300);
            }
        }
    })
    .catch(err => {
        console.error("Auto scan error:", err);
        autoScanActive = false; 
        const btn = document.getElementById(`auto-btn-${camId}`);
        btn.innerText = "Auto Scan: Error";
        btn.style.backgroundColor = "#7f8c8d";
    });
}