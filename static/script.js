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
let lastFrameTime = 0;
let frameTimes = [];
let lastDetectionTime = 0;
let roadDetectionEnabled = false;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    video = document.getElementById('video');
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        updateStatus('Camera API not supported in this browser', 'error');
    }
    
    // Add animation to stats cards on load
    animateStats();
    
    // Add event listener for road detection toggle
    document.getElementById('roadDetectionToggle').addEventListener('change', function() {
        roadDetectionEnabled = this.checked;
        if (isDetecting) {
            const mode = roadDetectionEnabled ? 'road detection' : 'normal';
            updateStatus(`<i class="fas fa-sync fa-spin"></i> Detection active (${mode} mode)`, 'success');
        }
    });
});

/**
 * Animate stats cards on page load
 */
function animateStats() {
    const statCards = document.querySelectorAll('.stat-card');
    statCards.forEach((card, index) => {
        setTimeout(() => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'all 0.5s ease';
            
            setTimeout(() => {
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 100);
        }, index * 200);
    });
}

/**
 * Start the device camera
 */
async function startCamera() {
    try {
        updateStatus('<i class="fas fa-spinner fa-spin"></i> Requesting camera access...', 'info');
        
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
            updateStatus('<i class="fas fa-check-circle"></i> Camera active - Ready for detection', 'success');
            
            // Update button states
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').classList.add('btn-disabled');
            document.getElementById('detectBtn').disabled = false;
            document.getElementById('detectBtn').classList.remove('btn-disabled');
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stopBtn').classList.remove('btn-disabled');
            
            // Remove pulse animation from start button
            document.getElementById('startBtn').classList.remove('pulse');
        };
        
    } catch (error) {
        console.error('Error accessing camera:', error);
        updateStatus(`<i class="fas fa-exclamation-triangle"></i> Camera access denied: ${error.message}`, 'error');
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
    
    updateStatus('<i class="fas fa-camera"></i> Camera stopped - Click "Start Camera" to begin', 'info');
    resetStats();
    
    // Add pulse animation back to start button
    document.getElementById('startBtn').classList.add('pulse');
}

/**
 * Toggle detection on/off
 */
function toggleDetection() {
    if (!isCameraActive) return;
    
    isDetecting = !isDetecting;
    const detectBtn = document.getElementById('detectBtn');
    
    if (isDetecting) {
        const mode = roadDetectionEnabled ? 'road detection' : 'normal';
        detectBtn.innerHTML = '<i class="fas fa-stop-circle"></i> Stop Detection';
        detectBtn.classList.remove('btn-secondary');
        detectBtn.classList.add('btn-danger');
        updateStatus(`<i class="fas fa-sync fa-spin"></i> Detection active (${mode} mode)`, 'success');
        
        // Start detection loop (5 FPS for better performance)
        lastDetectionTime = 0;
        detectionInterval = setInterval(captureAndDetect, 200);
        
        // Start FPS counter
        fpsInterval = setInterval(updateFPS, 1000);
        
        // Add pulse animation to detect button
        detectBtn.classList.add('pulse');
        
    } else {
        detectBtn.innerHTML = '<i class="fas fa-search"></i> Start Detection';
        detectBtn.classList.remove('btn-danger');
        detectBtn.classList.add('btn-secondary');
        updateStatus('<i class="fas fa-pause-circle"></i> Detection paused', 'warning');
        
        // Stop detection loop
        clearInterval(detectionInterval);
        clearInterval(fpsInterval);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Remove pulse animation
        detectBtn.classList.remove('pulse');
    }
}

/**
 * Capture frame and send to backend for detection
 */
async function captureAndDetect() {
    if (!isCameraActive || !isDetecting) return;
    
    const now = Date.now();
    // Limit detection to at most 5 times per second
    if (lastDetectionTime && (now - lastDetectionTime) < 200) {
        return;
    }
    lastDetectionTime = now;
    
    try {
        // Record frame time for FPS calculation
        if (lastFrameTime) {
            frameTimes.push(now - lastFrameTime);
            if (frameTimes.length > 10) frameTimes.shift(); // Keep last 10 frame times
        }
        lastFrameTime = now;
        
        // Create a temporary canvas to capture the current frame
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = video.videoWidth;
        tempCanvas.height = video.videoHeight;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(video, 0, 0);
        
        // Convert to blob with lower quality for faster transfer
        tempCanvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');
            
            // Add road detection parameter
            const roadDetectionParam = roadDetectionEnabled ? '?road_detection=true' : '';
            
            // Send to backend with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            try {
                const response = await fetch(`/detect${roadDetectionParam}`, {
                    method: 'POST',
                    body: formData,
                    signal: controller.signal
                });
                
                clearTimeout(timeoutId);
                
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                
                const data = await response.json();
                
                // Draw detections with enhanced visualization
                drawEnhancedDetections(data.detections, data.road_detection_enabled);
                
                // Update stats
                frameCount++;
                if (data.count > 0) {
                    detectionCount = data.count;
                    const avgConf = data.detections.reduce((sum, det) => sum + det.confidence, 0) / data.count;
                    totalConfidence = avgConf;
                } else if (data.road_detection_enabled && data.detections.length === 0) {
                    // Show a message when road detection is enabled but no road is detected
                    showNoRoadDetectedMessage();
                }
                
            } catch (error) {
                if (error.name === 'AbortError') {
                    console.warn('Detection request timed out');
                    updateStatus('<i class="fas fa-exclamation-circle"></i> Detection timeout - trying again...', 'warning');
                } else {
                    throw error;
                }
            }
            
        }, 'image/jpeg', 0.7); // Reduced quality for faster transfer
        
    } catch (error) {
        console.error('Detection error:', error);
        updateStatus(`<i class="fas fa-exclamation-circle"></i> Detection error: ${error.message}`, 'error');
    }
}

/**
 * Draw enhanced bounding boxes and labels on canvas
 */
function drawEnhancedDetections(detections, roadDetectionEnabled = false) {
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // If no detections and road detection is enabled, show a message
    if (!detections || detections.length === 0) {
        if (roadDetectionEnabled) {
            showNoRoadDetectedMessage();
        }
        return;
    }
    
    detections.forEach(detection => {
        const [x1, y1, x2, y2] = detection.bbox;
        const confidence = detection.confidence;
        const label = detection.label;
        
        // Calculate box dimensions
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Draw semi-transparent fill
        ctx.fillStyle = 'rgba(247, 37, 133, 0.2)';
        ctx.fillRect(x1, y1, width, height);
        
        // Draw bounding box with gradient
        const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
        gradient.addColorStop(0, '#f72585');
        gradient.addColorStop(1, '#4361ee');
        
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 3;
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw corner accents for a more modern look
        const cornerLength = Math.min(width, height) * 0.2;
        ctx.strokeStyle = '#4cc9f0';
        ctx.lineWidth = 4;
        
        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(x1, y1 + cornerLength);
        ctx.lineTo(x1, y1);
        ctx.lineTo(x1 + cornerLength, y1);
        ctx.stroke();
        
        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(x2 - cornerLength, y1);
        ctx.lineTo(x2, y1);
        ctx.lineTo(x2, y1 + cornerLength);
        ctx.stroke();
        
        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(x1, y2 - cornerLength);
        ctx.lineTo(x1, y2);
        ctx.lineTo(x1 + cornerLength, y2);
        ctx.stroke();
        
        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(x2 - cornerLength, y2);
        ctx.lineTo(x2, y2);
        ctx.lineTo(x2, y2 - cornerLength);
        ctx.stroke();
        
        // Draw label background with rounded corners
        const text = `${label} ${(confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 16px Inter, sans-serif';
        const textWidth = ctx.measureText(text).width;
        
        // Draw pill-shaped label background
        const labelX = x1 + 5;
        const labelY = y1 - 35;
        const labelWidth = textWidth + 20;
        const labelHeight = 30;
        const radius = 15;
        
        // Draw rounded rectangle
        ctx.beginPath();
        ctx.moveTo(labelX + radius, labelY);
        ctx.lineTo(labelX + labelWidth - radius, labelY);
        ctx.quadraticCurveTo(labelX + labelWidth, labelY, labelX + labelWidth, labelY + radius);
        ctx.lineTo(labelX + labelWidth, labelY + labelHeight - radius);
        ctx.quadraticCurveTo(labelX + labelWidth, labelY + labelHeight, labelX + labelWidth - radius, labelY + labelHeight);
        ctx.lineTo(labelX + radius, labelY + labelHeight);
        ctx.quadraticCurveTo(labelX, labelY + labelHeight, labelX, labelY + labelHeight - radius);
        ctx.lineTo(labelX, labelY + radius);
        ctx.quadraticCurveTo(labelX, labelY, labelX + radius, labelY);
        ctx.closePath();
        
        // Fill with gradient
        const labelGradient = ctx.createLinearGradient(labelX, labelY, labelX, labelY + labelHeight);
        labelGradient.addColorStop(0, '#4361ee');
        labelGradient.addColorStop(1, '#3a0ca3');
        ctx.fillStyle = labelGradient;
        ctx.fill();
        
        // Draw label text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, labelX + 10, labelY + 20);
        
        // Draw confidence meter
        if (confidence > 0.5) {
            const meterWidth = width * 0.6;
            const meterHeight = 6;
            const meterX = x1 + (width - meterWidth) / 2;
            const meterY = y2 + 10;
            
            // Background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(meterX, meterY, meterWidth, meterHeight);
            
            // Fill based on confidence
            let meterColor;
            if (confidence > 0.8) {
                meterColor = '#4ade80'; // Green
            } else if (confidence > 0.6) {
                meterColor = '#facc15'; // Yellow
            } else {
                meterColor = '#f72585'; // Red
            }
            
            ctx.fillStyle = meterColor;
            ctx.fillRect(meterX, meterY, meterWidth * confidence, meterHeight);
        }
    });
}

/**
 * Show a message when no road is detected in road detection mode
 */
function showNoRoadDetectedMessage() {
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw a semi-transparent overlay
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw warning message
    ctx.font = 'bold 24px Inter, sans-serif';
    ctx.fillStyle = '#facc15';
    ctx.textAlign = 'center';
    ctx.fillText('No Road Detected', canvas.width / 2, canvas.height / 2 - 20);
    
    ctx.font = '16px Inter, sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.fillText('Try pointing the camera at a road surface', canvas.width / 2, canvas.height / 2 + 20);
    ctx.fillText('or switch to normal detection mode', canvas.width / 2, canvas.height / 2 + 50);
    
    ctx.textAlign = 'left'; // Reset text alignment
}

/**
 * Update FPS counter with smoother calculation
 */
function updateFPS() {
    // Calculate average FPS from recent frame times
    if (frameTimes.length > 0) {
        const avgFrameTime = frameTimes.reduce((a, b) => a + b, 0) / frameTimes.length;
        const fps = Math.round(1000 / avgFrameTime);
        document.getElementById('fps').innerHTML = `${fps} <span style="font-size: 1rem;">FPS</span>`;
    } else {
        document.getElementById('fps').innerHTML = `${frameCount} <span style="font-size: 1rem;">FPS</span>`;
    }
    
    frameCount = 0;
    frameTimes = [];
    lastFrameTime = 0;
}

/**
 * Update status message with icon support
 */
function updateStatus(message, type = 'info') {
    const statusEl = document.getElementById('status');
    statusEl.innerHTML = message;
    statusEl.className = `status status-${type}`;
}

/**
 * Reset statistics
 */
function resetStats() {
    frameCount = 0;
    detectionCount = 0;
    totalConfidence = 0;
    frameTimes = [];
    lastFrameTime = 0;
    lastDetectionTime = 0;
    
    document.getElementById('detectionCount').textContent = '0';
    document.getElementById('fps').innerHTML = '0 <span style="font-size: 1rem;">FPS</span>';
    document.getElementById('confidence').textContent = '0%';
}

/**
 * Update statistics display with animation
 */
setInterval(() => {
    if (isDetecting) {
        // Animate detection count change
        const currentCountEl = document.getElementById('detectionCount');
        const currentValue = parseInt(currentCountEl.textContent);
        if (currentValue !== detectionCount) {
            currentCountEl.style.transform = 'scale(1.2)';
            setTimeout(() => {
                currentCountEl.textContent = detectionCount;
                currentCountEl.style.transform = 'scale(1)';
            }, 150);
        }
        
        // Update confidence
        document.getElementById('confidence').textContent = 
            totalConfidence > 0 ? `${(totalConfidence * 100).toFixed(0)}%` : '0%';
    }
}, 500);

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Spacebar to toggle detection
    if (e.code === 'Space' && isCameraActive) {
        e.preventDefault();
        toggleDetection();
    }
    
    // Escape to stop camera
    if (e.code === 'Escape' && isCameraActive) {
        stopCamera();
    }
});