"""
FastAPI Backend for Pothole Detection
This server provides an endpoint for real-time pothole detection using YOLOv8.
"""

import base64
import io
import logging
import random
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Pothole Detector API", version="1.0.0")

# Add CORS middleware to allow browser requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variable to store detection status
model_loaded = True  # Simulated - always ready

@app.on_event("startup")
async def load_model():
    """Initialize detection system"""
    global model_loaded
    try:
        logger.info("Initializing pothole detection system...")
        logger.info("NOTE: Using simulated detection for MVP demo")
        logger.info("To use real YOLOv8 model, install: pip install torch torchvision ultralytics")
        model_loaded = True
        logger.info("Detection system ready!")
    except Exception as e:
        logger.error(f"Error initializing detection: {e}")
        raise

def detect_with_edge_detection(img_array):
    """
    Simulated pothole detection using OpenCV edge detection
    This is a placeholder for demonstration purposes
    Replace with actual YOLOv8 model for production use
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply moderate blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Balanced edge detection
    edges = cv2.Canny(blurred, 60, 180)
    
    # Light morphological operations to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    height, width = img_array.shape[:2]
    
    # Filter and convert contours to bounding boxes with balanced criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Balanced area filtering - not too strict
        if area < 600 or area > (width * height * 0.25):
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio to filter out very vertical gaps (like trees)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Filter out very tall/narrow shapes (like gaps between trees)
        # But allow more variation for actual potholes
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue
        
        # Less strict size filtering
        if w < 20 or h < 20:  # Too small
            continue
        
        # More lenient position filter - focus on middle to bottom area
        # Only skip very top portion (likely sky)
        if y < height * 0.15:
            continue
        
        # Calculate confidence based on shape and position
        shape_score = 1 - abs(1 - aspect_ratio) if aspect_ratio <= 2 else 0.5
        size_score = min(area / 3000, 1.0)
        position_score = (y / height)  # Lower in image = higher score
        
        confidence = (shape_score * 0.4 + size_score * 0.3 + position_score * 0.3) * random.uniform(0.65, 0.95)
        
        # Lower confidence threshold to allow more detections
        if confidence < 0.35:
            continue
        
        detections.append({
            "label": "pothole",
            "confidence": round(confidence, 2),
            "bbox": [x, y, x + w, y + h]
        })
    
    # Show top 5 detections
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections[:5]

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    try:
        with open("index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Pothole Detector API</h1><p>Frontend not found. Please ensure index.html exists.</p>")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post("/detect")
async def detect_potholes(file: UploadFile = File(...)) -> Dict:
    """
    Detect potholes in an uploaded image
    
    Args:
        file: Image file (JPEG/PNG) or base64 encoded image
        
    Returns:
        JSON with detection results including bounding boxes and confidence scores
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    try:
        # Read the image
        contents = await file.read()
        
        # Handle base64 encoded images
        if contents.startswith(b'data:image'):
            # Extract base64 data
            base64_data = contents.split(b',')[1]
            contents = base64.b64decode(base64_data)
        
        # Convert to PIL Image
        image = Image.open(io.BytesIO(contents))
        
        # Convert to numpy array (RGB format)
        img_array = np.array(image)
        
        # Ensure RGB format
        if len(img_array.shape) == 2:  # Grayscale
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Run detection (simulated using edge detection)
        detections = detect_with_edge_detection(img_array)
        
        return {
            "detections": detections,
            "count": len(detections),
            "image_size": {"width": image.width, "height": image.height}
        }
        
    except Exception as e:
        logger.error(f"Error during detection: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the detection system"""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Detection system not initialized")
    
    return {
        "detection_method": "OpenCV Edge Detection (MVP Demo)",
        "note": "This is a simulated detection for demonstration. For production, install YOLOv8: pip install torch torchvision ultralytics",
        "recommendation": "Replace with a pothole-trained YOLOv8 model from Roboflow or Kaggle for real-world use"
    }

# Mount static files for JavaScript and CSS
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
