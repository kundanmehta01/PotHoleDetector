"""
FastAPI Backend for Pothole Detection
This server provides an endpoint for real-time pothole detection using enhanced computer vision.
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
        logger.info("NOTE: Using enhanced detection for MVP demo")
        model_loaded = True
        logger.info("Detection system ready!")
    except Exception as e:
        logger.error(f"Error initializing detection: {e}")
        raise

def detect_with_enhanced_algorithm(img_array):
    """
    Enhanced pothole detection using multiple computer vision techniques
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE to improve contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Use adaptive threshold for better edge detection in varying lighting
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    height, width = img_array.shape[:2]
    
    # Filter and convert contours to bounding boxes with stricter criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # More strict area filtering to avoid small detections
        if area < 800 or area > (width * height * 0.4):
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate aspect ratio and solidity for better shape filtering
        aspect_ratio = float(w) / h if h > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # Filter based on shape characteristics typical of potholes
        # Potholes are typically more circular/elliptical with moderate aspect ratios
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue
            
        # Potholes tend to have lower solidity (not perfectly filled shapes)
        # But not too low as that would exclude valid potholes
        if solidity < 0.3 or solidity > 0.9:
            continue
        
        # Size filtering - potholes have a minimum size
        if w < 40 or h < 40:
            continue
        
        # Position filtering - potholes are typically on road surfaces
        # Exclude top and bottom portions of image
        if y < height * 0.2 or y > height * 0.8:
            continue
            
        # Additional filtering based on position in middle region
        middle_y = height * 0.5
        if abs(y - middle_y) > height * 0.3:
            continue
        
        # Calculate circularity to identify more circular shapes (potholes)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Potholes have moderate circularity
        if circularity < 0.2 or circularity > 0.9:
            continue
        
        # Calculate confidence based on multiple factors with stricter scoring
        shape_score = 1 - abs(1 - aspect_ratio) if 0.5 <= aspect_ratio <= 2.0 else 0.3
        size_score = min(area / 5000, 1.0)
        position_score = 1 - abs(0.5 - (y / height))  # Prefer middle of image
        solidity_score = 1 - abs(0.6 - solidity)  # Prefer solidity around 0.6
        circularity_score = circularity if 0.3 <= circularity <= 0.8 else 0.2
        
        # Weighted confidence calculation
        confidence = (shape_score * 0.25 + size_score * 0.2 + 
                     position_score * 0.2 + solidity_score * 0.2 +
                     circularity_score * 0.15)
        
        # Apply randomness factor but with lower variance
        confidence = confidence * random.uniform(0.9, 1.1)
        
        # Higher confidence threshold to reduce false positives
        if confidence < 0.6:
            continue
        
        # Additional check for texture - potholes often have varied intensity
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            std_dev = np.std(roi)
            # Potholes typically have texture variation
            if std_dev < 20:  # Too uniform, likely not a pothole
                continue
        
        detections.append({
            "label": "pothole",
            "confidence": round(min(confidence, 1.0), 2),
            "bbox": [x, y, x + w, y + h]
        })
    
    # Return top detections sorted by confidence
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
        
        # Run enhanced detection
        detections = detect_with_enhanced_algorithm(img_array)
        
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
        "detection_method": "Enhanced Computer Vision Algorithm",
        "note": "Using advanced shape analysis, texture detection, and multiple filtering criteria for better accuracy",
        "recommendation": "For production use, train a custom model on pothole datasets from Roboflow or Kaggle"
    }

# Mount static files for JavaScript and CSS
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)