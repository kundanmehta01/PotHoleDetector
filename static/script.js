/**
 * Pothole Detector - Frontend JavaScript
 * Handles camera access, frame capture, and real-time detection visualization
 */

let video, canvas, ctx;
let isDetecting = false;
let isCameraActive = false;
let stream = null;
let detectionInterval = null;
let fpsInterval = null;
let frameCount = 0;
let totalConfidence = 0;
let detectionCount = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateStatus('Camera API not supported in this browser', 'error');
    }
});

/**
 * Start the device camera
 */
async function startCamera() {
    try {
        updateStatus('Requesting camera access...', 'info');
        
        // Request camera access
        // Use facingMode: environment for rear camera on mobile, user for front camera
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: 'environment' // Use rear camera on mobile
            }
        };
        
        stream = await navigator.mediaDevices.getUserMedia(constraints);
        video.srcObject = stream;
        
        video.onloadedmetadata = () => {
            // Set canvas size to match video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            
            isCameraActive = true;
            updateStatus('Camera active - Ready for detection', 'success');
            
            // Update button states
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').classList.add('btn-disabled');
            document.getElementById('detectBtn').disabled = false;
            document.getElementById('detectBtn').classList.remove('btn-disabled');
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stopBtn').classList.remove('btn-disabled');
        };
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        updateStatus(`Camera access denied: ${error.message}`, 'error');
    }
}

/**
 * Stop the camera stream
 */
function stopCamera() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        stream = null;
    }
    
    if (isDetecting) {
        toggleDetection();
    }
    
    isCameraActive = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Reset button states
    document.getElementById('startBtn').disabled = false;
    document.getElementById('startBtn').classList.remove('btn-disabled');
    document.getElementById('detectBtn').disabled = true;
    document.getElementById('detectBtn').classList.add('btn-disabled');
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('stopBtn').classList.add('btn-disabled');
    
    updateStatus('Camera stopped', 'info');
    resetStats();
}

/**
 * Toggle detection on/off
 */
function toggleDetection() {
    if (!isCameraActive) return;
    
    isDetecting = !isDetecting;
    const detectBtn = document.getElementById('detectBtn');
    
    if (isDetecting) {
        detectBtn.textContent = 'Stop Detection';
        detectBtn.classList.remove('btn-secondary');
        detectBtn.classList.add('btn-danger');
        updateStatus('Detection active - Processing frames...', 'success');
        
        // Start detection loop (3 FPS for better performance)
        detectionInterval = setInterval(captureAndDetect, 333);
        
        // Start FPS counter
        fpsInterval = setInterval(updateFPS, 1000);
        
    } else {
        detectBtn.textContent = 'Start Detection';
        detectBtn.classList.remove('btn-danger');
        detectBtn.classList.add('btn-secondary');
        updateStatus('Detection paused', 'warning');
        
        // Stop detection loop
        clearInterval(detectionInterval);
        clearInterval(fpsInterval);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

/**
 * Capture frame and send to backend for detection
 */
async function captureAndDetect() {
    if (!isCameraActive || !isDetecting) return;
    
    try {
        // Create a temporary canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to blob
        tempCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            
            // Send to backend
            const response = await fetch('/detect', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Draw detections
            drawDetections(data.detections);
            
            // Update stats
            frameCount++;
            if (data.count > 0) {
                detectionCount = data.count;
                const avgConf = data.detections.reduce((sum, det) => sum + det.confidence, 0) / data.count;
                totalConfidence = avgConf;
            }
            
        }, 'image/jpeg', 0.8);
        
    } catch (error) {
        console.error('Detection error:', error);
        updateStatus(`Detection error: ${error.message}`, 'error');
    }
}

/**
 * Draw bounding boxes and labels on canvas
 */
function drawDetections(detections) {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!detections || detections.length === 0) return;
    
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = detection.confidence;
        const label = detection.label;
        
        // Draw bounding box
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw label background
        const text = `${label} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px Arial';
        const textWidth = ctx.measureText(text).width;
        
        ctx.fillStyle = '#ff6b6b';
        ctx.fillRect(x1, y1 - 30, textWidth + 10, 25);
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 5, y1 - 10);
    });
}

/**
 * Update FPS counter
 */
function updateFPS() {
    document.getElementById('fps').textContent = frameCount;
    frameCount = 0;
}

/**
 * Update status message
 */
function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('status');
    statusEl.textContent = message;
    statusEl.className = `status status-${type}`;
}

/**
 * Reset statistics
 */
function resetStats() {
    frameCount = 0;
    detectionCount = 0;
    totalConfidence = 0;
    
    document.getElementById('detectionCount').textContent = '0';
    document.getElementById('fps').textContent = '0';
    document.getElementById('confidence').textContent = '0%';
}

/**
 * Update statistics display
 */
setInterval(() => {
    if (isDetecting) {
        document.getElementById('detectionCount').textContent = detectionCount;
        document.getElementById('confidence').textContent = 
            totalConfidence > 0 ? `${(totalConfidence * 100).toFixed(0)}%` : '0%';
    }
}, 500);
