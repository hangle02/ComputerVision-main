// --- ORIGINAL APP LOGIC ---
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
            revertToPlaceholders(camId);
        }
    }).catch(err => console.error(err));
}

function captureAndProcess(camId, stepName) {
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
            document.getElementById(`fragment-${camId}`).src = data.processed;
            document.getElementById(`proc-time-${camId}`).innerText = `Process time: ${data.process_time_ms} ms`;
            
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
        alert('Network error occurred.');
        console.error(err);
    });
}

function rotateImage(camId) {
    const imgElement = document.getElementById(`fragment-${camId}`);
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
            document.getElementById(`proc-time-${camId}`).innerText = "Rotated successfully!";
            setTimeout(() => { document.getElementById(`proc-time-${camId}`).innerText = originalTimeText; }, 2000);
        } else {
            alert('Error rotating image: ' + data.error);
        }
    }).catch(err => console.error(err));
}

// --- PLACEHOLDER LOGIC ---
function initPlaceholders(camId = 1) {
    const video = document.getElementById(`video-${camId}`);
    if (video) { video.onerror = function() { handleBrokenStream(this); }; }
}

function handleBrokenStream(img) {
    const streamPlaceholder = '/static/images/default-stream.gif';
    if (img.src && !img.src.includes(streamPlaceholder)) {
        img.removeAttribute('src'); 
        img.src = streamPlaceholder; 
    }
}

function revertToPlaceholders(camId) {
    document.getElementById(`video-${camId}`).src = '/static/images/default-stream.gif';
    document.getElementById(`captured-${camId}`).src = '/static/images/default-capture.png';
    document.getElementById(`fragment-${camId}`).src = '/static/images/default-processed.png';
    document.getElementById(`proc-time-${camId}`).innerText = 'Process time: -- ms';
    const matrixBox = document.getElementById(`matrix-display-${camId}`);
    if (matrixBox) matrixBox.style.display = 'none';
}

// --- PWA INSTALLATION LOGIC ---
let deferredPrompt;
const installBtn = document.getElementById('install-btn');

window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn.style.display = 'block';
});

installBtn.addEventListener('click', async () => {
    if (!deferredPrompt) return;
    deferredPrompt.prompt();
    const { outcome } = await deferredPrompt.userChoice;
    deferredPrompt = null;
    installBtn.style.display = 'none';
});

window.addEventListener('appinstalled', () => {
    installBtn.style.display = 'none';
    deferredPrompt = null;
});

// --- PWA SERVICE WORKER REGISTRATION ---
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js')
      .then(reg => console.log('Service Worker registered!'))
      .catch(err => console.error('Service Worker failed: ', err));
  });
}

// Initialize Placeholders on load
document.addEventListener('DOMContentLoaded', () => {
    initPlaceholders(1);
});