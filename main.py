"""
FastAPI Backend for Pothole Detection
This server provides an endpoint for real-time pothole detection using optimized computer vision.
"""

import base64
import io
import logging
import random
from typing import List, Dict
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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
        logger.info("Initializing optimized pothole detection system...")
        model_loaded = True
        logger.info("Detection system ready!")
    except Exception as e:
        logger.error(f"Error initializing detection: {e}")
        raise

def detect_road_region(img_array):
    """
    Detect road region in the image using color and texture analysis
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    
    # Define range for typical road colors (grays, browns, blacks)
    # These values might need adjustment based on lighting conditions
    lower_road = np.array([0, 0, 30])
    upper_road = np.array([180, 50, 200])
    
    # Create mask for road-like colors
    road_mask = cv2.inRange(hsv, lower_road, upper_road)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
    
    # Find the largest connected component (likely the road)
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Only consider it a road if it's large enough
    if cv2.contourArea(largest_contour) < img_array.shape[0] * img_array.shape[1] * 0.3:
        return None
    
    # Get bounding rectangle of the road region
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Return the road region coordinates
    return (x, y, x + w, y + h)

def detect_with_optimized_algorithm(img_array, road_detection=False):
    """
    Optimized pothole detection using efficient computer vision techniques
    """
    # If road detection is enabled, first identify road region
    road_region = None
    if road_detection:
        road_region = detect_road_region(img_array)
        # If no road is detected, return empty detections
        if road_region is None:
            return []
    
    # Resize image for faster processing while maintaining aspect ratio
    height, width = img_array.shape[:2]
    max_dimension = 640  # Reduce processing load
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_array = cv2.resize(img_array, (new_width, new_height))
        
        # Scale road region coordinates if they exist
        if road_region:
            rx1, ry1, rx2, ry2 = road_region
            scale_x = new_width / width
            scale_y = new_height / height
            road_region = (
                int(rx1 * scale_x), 
                int(ry1 * scale_y), 
                int(rx2 * scale_x), 
                int(ry2 * scale_y)
            )
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Apply CLAHE to improve contrast (faster version)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur for noise reduction (faster than bilateral)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Edge detection with optimized parameters
    edges = cv2.Canny(blurred, 50, 150)
    
    # If road region is specified, mask edges outside this region
    if road_region and road_detection:
        rx1, ry1, rx2, ry2 = road_region
        # Create mask for road region
        mask = np.zeros_like(edges)
        mask[ry1:ry2, rx1:rx2] = 255
        # Apply mask to edges
        edges = cv2.bitwise_and(edges, edges, mask=mask)
    
    # Morphological operations to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detections = []
    
    # Filter and convert contours to bounding boxes with balanced criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Adjusted area filtering for better balance
        if area < 400 or area > (img_array.shape[1] * img_array.shape[0] * 0.3):
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # If road detection is enabled, check if pothole is within road region
        if road_region and road_detection:
            rx1, ry1, rx2, ry2 = road_region
            # Check if bounding box is within road region
            if x < rx1 or y < ry1 or (x + w) > rx2 or (y + h) > ry2:
                continue
        
        # Calculate aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Filter based on shape characteristics with more lenient criteria
        if aspect_ratio < 0.3 or aspect_ratio > 4.0:
            continue
        
        # Size filtering
        if w < 30 or h < 30:
            continue
        
        # Position filtering - focus on likely road area
        if y < img_array.shape[0] * 0.1 or y > img_array.shape[0] * 0.9:
            continue
        
        # Calculate confidence with balanced scoring
        shape_score = 1 - abs(1 - aspect_ratio) if aspect_ratio <= 2 else 0.5
        size_score = min(area / 2000, 1.0)
        position_score = (y / img_array.shape[0])  # Lower in image = higher score
        
        confidence = (shape_score * 0.4 + size_score * 0.3 + position_score * 0.3) * random.uniform(0.8, 1.0)
        
        # Lowered threshold to detect more potholes
        if confidence < 0.4:
            continue
        
        # Scale coordinates back to original image size
        scale_x = width / img_array.shape[1]
        scale_y = height / img_array.shape[0]
        
        detections.append({
            "label": "pothole",
            "confidence": round(confidence, 2),
            "bbox": [int(x * scale_x), int(y * scale_y), int((x + w) * scale_x), int((y + h) * scale_y)]
        })
    
    # Return top detections sorted by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    return detections[:8]  # Increased from 5 to 8 to detect more potholes

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
async def detect_potholes(
    file: UploadFile = File(...),
    road_detection: bool = Query(False, description="Enable road region detection before pothole detection")
) -> Dict:
    """
    Detect potholes in an uploaded image
    
    Args:
        file: Image file (JPEG/PNG) or base64 encoded image
        road_detection: Enable road region detection before pothole detection
        
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
        
        # Run optimized detection
        detections = detect_with_optimized_algorithm(img_array, road_detection)
        
        return {
            "detections": detections,
            "count": len(detections),
            "road_detection_enabled": road_detection,
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
        "detection_method": "Optimized Computer Vision Algorithm",
        "road_detection": "Available - Detects road regions before pothole detection",
        "note": "Using efficient image processing techniques for real-time performance",
        "recommendation": "For production use, train a custom model on pothole datasets from Roboflow or Kaggle"
    }

# Mount static files for JavaScript and CSS
app.mount("/static", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)