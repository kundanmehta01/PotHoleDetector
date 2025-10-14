# Pothole Detector - Complete Setup Guide

## 🚀 Quick Start (MVP Demo)

The current implementation uses **OpenCV edge detection** for demonstration purposes and works immediately without additional setup.

### 1. Start the Server
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

### 2. Access the App
- **Local**: Open `http://localhost:5000` in your browser
- **Mobile/Other Device**: Use your computer's IP address (e.g., `http://192.168.1.100:5000`)

### 3. Use the App
1. Click **"Start Camera"** and allow camera access
2. Click **"Start Detection"** to begin real-time analysis
3. Watch detected objects appear with bounding boxes
4. Monitor FPS and confidence metrics

---

## 🔧 Production Setup with Real YOLOv8

### Prerequisites
- Python 3.11+
- ~3GB free disk space (for PyTorch + CUDA)
- Modern browser (Chrome, Firefox, Safari, Edge)
- Camera-enabled device

### Step 1: Install Dependencies

#### Option A: CPU-Only (Lighter, Slower)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

#### Option B: GPU-Enabled (Faster, Larger)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
```

### Step 2: Get a Pothole Detection Model

#### Option 1: Use Pre-trained Model from Roboflow
1. Visit [Roboflow Universe - Pothole Detection](https://universe.roboflow.com/search?q=pothole)
2. Find a pothole detection dataset
3. Export in **YOLOv8** format
4. Download the model weights (`.pt` file)

#### Option 2: Train Your Own Model
```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO('yolov8n.pt')

# Train on your pothole dataset
results = model.train(
    data='pothole_dataset/data.yaml',  # Your dataset config
    epochs=100,
    imgsz=640,
    batch=16
)

# Export the trained model
model.export(format='pytorch')
```

#### Option 3: Use Kaggle Dataset
1. Download pothole dataset from [Kaggle](https://www.kaggle.com/search?q=pothole+detection)
2. Prepare data in YOLO format
3. Train using Ultralytics YOLO

### Step 3: Update main.py

Replace the simulated detection with real YOLO inference:

```python
# At the top of main.py, uncomment:
from ultralytics import YOLO

# Global model variable
model = None

@app.on_event("startup")
async def load_model():
    """Load the YOLOv8 model on startup"""
    global model
    try:
        logger.info("Loading YOLOv8 pothole detection model...")
        # Load your trained pothole model
        model = YOLO('models/pothole_yolov8n.pt')
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Replace detect_with_edge_detection() with YOLO inference
@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)) -> Dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # ... image loading code ...
    
    # Run YOLO inference
    results = model(img_array, verbose=False)
    
    detections = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            detections.append({
                "label": class_name,
                "confidence": round(confidence, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)]
            })
    
    return {
        "detections": detections,
        "count": len(detections),
        "image_size": {"width": image.width, "height": image.height}
    }
```

### Step 4: Organize Model Files

Create a models directory:
```bash
mkdir models
mv your_pothole_model.pt models/pothole_yolov8n.pt
```

### Step 5: Test the Production Setup

1. Restart the server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000
   ```

2. Check model loading in logs:
   ```
   INFO: Loading YOLOv8 pothole detection model...
   INFO: Model loaded successfully!
   ```

3. Test with real pothole images

---

## 📱 Mobile/Remote Access

### Network Configuration

1. **Find your computer's IP**:
   ```bash
   # Linux/Mac
   ifconfig | grep "inet "
   
   # Windows
   ipconfig
   ```

2. **Access from mobile**:
   - Ensure devices are on same network
   - Visit `http://YOUR_IP:5000` on mobile browser
   - Allow camera access when prompted

3. **Use rear camera on mobile**:
   - The app automatically requests `facingMode: environment`
   - This selects the rear camera on mobile devices

### Firewall Settings

If you can't access from other devices:

```bash
# Linux (allow port 5000)
sudo ufw allow 5000

# Or run on different port
uvicorn main:app --host 0.0.0.0 --port 8080
```

---

## 🎯 Performance Optimization

### 1. Adjust Detection Frequency
In `static/script.js`:
```javascript
// Change from 3 FPS to 1 FPS for slower devices
detectionInterval = setInterval(captureAndDetect, 1000);  // 1000ms = 1 FPS
```

### 2. Reduce Image Resolution
In `static/script.js`:
```javascript
const constraints = {
    video: {
        width: { ideal: 640 },   // Lower from 1280
        height: { ideal: 480 },  // Lower from 720
        facingMode: 'environment'
    }
};
```

### 3. Use Smaller YOLO Model
```python
# Use nano model for speed
model = YOLO('yolov8n.pt')  # Fastest

# Or medium for better accuracy
model = YOLO('yolov8m.pt')  # Slower but more accurate
```

### 4. Batch Processing
For high-traffic scenarios, implement request queuing:
```python
from asyncio import Queue

detection_queue = Queue()

@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)):
    # Add to queue and process in batches
    await detection_queue.put(file)
    # ... batch processing logic
```

---

## 🐛 Troubleshooting

### Camera Not Working

**Issue**: Camera access denied
**Solution**:
- Ensure HTTPS or localhost (required for camera access)
- Check browser permissions
- Try different browser (Chrome works best)

### Model Not Loading

**Issue**: CUDA errors or memory issues
**Solution**:
- Install CPU-only PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Use smaller model: YOLOv8n instead of YOLOv8x

### Slow Detection

**Issue**: Low FPS or lag
**Solution**:
- Reduce camera resolution
- Lower detection frequency (1-2 FPS)
- Use GPU if available
- Switch to YOLOv8n (nano) model

### Port Already in Use

**Issue**: Port 5000 occupied
**Solution**:
```bash
# Use different port
uvicorn main:app --host 0.0.0.0 --port 8000

# Or kill existing process
lsof -ti:5000 | xargs kill -9
```

---

## 🔐 Security Considerations

### Production Deployment

1. **Enable HTTPS** (required for camera access on non-localhost):
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 5000 --ssl-keyfile key.pem --ssl-certfile cert.pem
   ```

2. **Add Authentication**:
   ```python
   from fastapi import Depends, HTTPException
   from fastapi.security import HTTPBearer
   
   security = HTTPBearer()
   
   @app.post("/detect")
   async def detect_potholes(
       file: UploadFile = File(...),
       credentials = Depends(security)
   ):
       # Verify token
       # ... detection logic
   ```

3. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   from slowapi.util import get_remote_address
   
   limiter = Limiter(key_func=get_remote_address)
   app.state.limiter = limiter
   
   @app.post("/detect")
   @limiter.limit("10/minute")
   async def detect_potholes(...):
       # ... detection logic
   ```

---

## 📊 Monitoring & Logging

### Enable Detailed Logging
```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pothole_detector.log'),
        logging.StreamHandler()
    ]
)
```

### Track Detection Metrics
```python
from prometheus_client import Counter, Histogram

detection_counter = Counter('pothole_detections_total', 'Total potholes detected')
detection_time = Histogram('detection_duration_seconds', 'Detection processing time')

@app.post("/detect")
async def detect_potholes(...):
    with detection_time.time():
        # ... detection logic
        detection_counter.inc(len(detections))
```

---

## 🌐 Deployment Options

### 1. Replit Deployment
- Click "Deploy" button in Replit
- App will be accessible via Replit domain
- HTTPS enabled automatically

### 2. Cloud Deployment (AWS/GCP/Azure)
- Use Docker container
- Deploy to VM or container service
- Configure HTTPS with Let's Encrypt

### 3. Docker Deployment
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
```

```bash
docker build -t pothole-detector .
docker run -p 5000:5000 pothole-detector
```

---

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Roboflow Pothole Datasets](https://universe.roboflow.com/search?q=pothole)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

## 💡 Next Steps

1. **Improve Detection**:
   - Train on larger pothole dataset
   - Fine-tune confidence thresholds
   - Add temporal filtering (track detections across frames)

2. **Add Features**:
   - GPS tagging for detected potholes
   - Database storage for detection history
   - Map visualization with heatmaps
   - Email/SMS alerts for new detections

3. **Optimize Performance**:
   - Implement model quantization
   - Use TensorRT for GPU inference
   - Add edge caching for repeat detections

4. **Scale the System**:
   - Multi-user support
   - Cloud-based processing
   - Mobile app integration
   - Real-time dashboard
