# Pothole Detector Web Application

## 📋 Project Overview
A complete MVP web application for real-time pothole detection using device cameras (laptop/phone/tablet). Built with FastAPI backend and browser-based frontend with live video streaming and detection visualization.

## 🎯 Current State (October 2025)
- **Status**: MVP Complete and Running
- **Detection Method**: OpenCV edge detection (lightweight demo)
- **Server**: Running on port 5000 with FastAPI + Uvicorn
- **Frontend**: Responsive HTML/CSS/JS with camera access and canvas overlay

## 🏗️ Project Architecture

### Backend (FastAPI)
- **File**: `main.py`
- **Framework**: FastAPI with CORS middleware
- **Detection**: OpenCV-based edge detection (simulated pothole detection)
- **Endpoints**:
  - `GET /` - Serves the main HTML interface
  - `POST /detect` - Processes frames and returns detection results
  - `GET /health` - Health check
  - `GET /model-info` - Detection system information
- **Image Processing**: PIL + NumPy + OpenCV

### Frontend
- **File**: `index.html`
- **JavaScript**: `static/script.js`
- **Features**:
  - Real-time camera access via getUserMedia API
  - Canvas-based bounding box overlay
  - Frame capture at ~3 FPS
  - Live statistics (FPS, detection count, confidence)
  - Responsive design for mobile/desktop

### Tech Stack
- **Backend**: Python 3.11, FastAPI, Uvicorn, OpenCV, NumPy, Pillow
- **Frontend**: Vanilla HTML5, CSS3, JavaScript (no frameworks)
- **Detection**: OpenCV Canny edge detection (MVP)

## 📦 Dependencies

### Installed Packages
```
fastapi==0.118.3
uvicorn[standard]==0.37.0
opencv-python-headless==4.12.0.88
numpy==2.2.6
python-multipart==0.0.20
pillow==11.3.0
```

### NOT Installed (Too Large for Disk Quota)
- `torch` - PyTorch (requires ~2GB with CUDA dependencies)
- `torchvision` - Vision models
- `ultralytics` - YOLOv8 (depends on PyTorch)

## 🚀 Setup & Deployment

### Running the App
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 5000
```

### Accessing the App
- Local: `http://localhost:5000`
- Mobile/Remote: `http://YOUR_IP:5000`

## 🔄 Upgrading to Real YOLOv8 Detection

The current implementation uses edge detection for demonstration. To use real pothole detection:

### Step 1: Install YOLOv8 Dependencies
```bash
# CPU-only version (lighter)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics

# OR GPU version (faster)
pip install torch torchvision ultralytics
```

### Step 2: Get Pothole Model
- Download from [Roboflow Universe](https://universe.roboflow.com/search?q=pothole)
- Or train your own using [Kaggle datasets](https://www.kaggle.com/search?q=pothole+detection)

### Step 3: Update main.py
- Replace `detect_with_edge_detection()` with YOLOv8 inference
- Load trained model: `model = YOLO('pothole_model.pt')`
- See `SETUP.md` for complete instructions

## 📁 File Structure
```
pothole-detector/
├── main.py              # FastAPI backend server
├── index.html           # Frontend UI
├── static/
│   └── script.js        # Camera and detection logic
├── requirements.txt     # Python dependencies
├── README.md           # User documentation
├── SETUP.md            # Detailed setup guide
├── .gitignore          # Git ignore patterns
└── replit.md           # This file (project memory)
```

## 🎨 Design Decisions

### Why OpenCV Instead of YOLOv8 (MVP)?
- **Disk Space**: PyTorch + CUDA dependencies exceed Replit disk quota (~3GB needed)
- **Demonstration**: Edge detection shows the app flow and UI functionality
- **Upgradeable**: Architecture supports easy swap to YOLOv8 when deployed elsewhere

### Why Vanilla JavaScript?
- **Simplicity**: No build tools or frameworks needed
- **Performance**: Direct DOM manipulation for real-time rendering
- **Compatibility**: Works on all modern browsers
- **Lightweight**: Minimal dependencies

### Detection Flow
1. Browser captures video frame via `getUserMedia()`
2. Canvas element draws frame and converts to blob
3. Frame sent as multipart/form-data to `/detect` endpoint
4. Backend processes with OpenCV edge detection
5. Returns JSON with bounding boxes and confidence scores
6. Frontend draws boxes on canvas overlay

## 🔧 Configuration

### Camera Settings (static/script.js)
```javascript
const constraints = {
    video: {
        width: { ideal: 1280 },
        height: { ideal: 720 },
        facingMode: 'environment'  // Rear camera on mobile
    }
};
```

### Detection Frequency
```javascript
detectionInterval = setInterval(captureAndDetect, 333);  // ~3 FPS
```

### Edge Detection Parameters (main.py)
```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
# Min area: 500px, Max: 30% of frame
```

## 🐛 Known Issues & Limitations

### Current Limitations
1. **Detection Accuracy**: Edge detection is not real pothole detection
2. **False Positives**: Any edge/contour may be detected
3. **No Persistence**: Detections don't persist across frames
4. **No Database**: No storage of detection history

### Browser Compatibility
- ✅ Chrome/Edge (best performance)
- ✅ Firefox
- ✅ Safari (iOS 11+)
- ⚠️ Requires HTTPS for camera access (except localhost)

## 🚀 Future Enhancements

### Phase 1: Real Detection
- [ ] Install YOLOv8 with pothole-trained model
- [ ] Replace edge detection with proper object detection
- [ ] Add confidence thresholding
- [ ] Implement non-maximum suppression

### Phase 2: Data Persistence
- [ ] Add PostgreSQL database
- [ ] Store detection events with timestamps
- [ ] GPS tagging for location data
- [ ] Detection history API

### Phase 3: Visualization
- [ ] Interactive map with Leaflet.js
- [ ] Pothole density heatmap
- [ ] Detection statistics dashboard
- [ ] Export reports (PDF/CSV)

### Phase 4: Scale
- [ ] Multi-user support
- [ ] Authentication & authorization
- [ ] Cloud storage for images
- [ ] Real-time WebSocket updates
- [ ] Mobile app wrapper

## 📝 Development Notes

### Running Workflow
- **Name**: Server
- **Command**: `python -m uvicorn main:app --host 0.0.0.0 --port 5000`
- **Port**: 5000 (webview)
- **Status**: ✅ Running

### Testing Checklist
- [x] Server starts without errors
- [x] Frontend loads correctly
- [x] Camera access works
- [x] Frame capture and send to backend
- [x] Detection returns results
- [x] Bounding boxes render on canvas
- [x] FPS counter updates
- [x] Responsive on mobile

### Git Ignore
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`.pythonlibs/`, `venv/`)
- Model weights (`*.pt`)
- Environment variables (`.env`)
- IDE files (`.vscode/`, `.idea/`)

## 🤝 User Preferences
- Clean, production-ready code with comments
- Modular structure for easy upgrades
- Comprehensive documentation
- Security best practices

## 📚 Documentation Files
- `README.md` - User-facing documentation with quick start
- `SETUP.md` - Comprehensive setup guide with YOLOv8 integration
- `replit.md` - This file (project memory and architecture)

## 🔗 Key Resources
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [YOLOv8 Docs](https://docs.ultralytics.com/)
- [Roboflow Pothole Datasets](https://universe.roboflow.com/search?q=pothole)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

**Last Updated**: October 10, 2025
**Version**: 1.0.0 (MVP)
**Status**: ✅ Complete and Running
