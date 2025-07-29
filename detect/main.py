from fastapi import FastAPI, UploadFile, File, Query
from typing import List, Optional
from batch_detect import process_image, read_image
import uvicorn
import cv2
import numpy as np
import time
import os
from datetime import datetime

app = FastAPI(
    title="CV Detection API",
    description="Object detection and OCR for food packaging (ingredients + nutrition) with smart confidence boosting",
    version="2.0.0"
)

def capture_camera_image():
    """Capture image from camera"""
    camera = None
    try:
        # Try different camera configurations
        camera_configs = [
            (0, cv2.CAP_ANY),
            (0, cv2.CAP_DSHOW),
            (1, cv2.CAP_ANY),
            (1, cv2.CAP_DSHOW),
        ]
        
        for camera_index, backend in camera_configs:
            try:
                print(f"🔍 Trying camera {camera_index} with backend {backend}")
                camera = cv2.VideoCapture(camera_index, backend)
                
                if camera.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = camera.read()
                    if ret and test_frame is not None:
                        print(f"✅ Camera {camera_index} working with backend {backend}")
                        break
                    else:
                        print(f"❌ Camera {camera_index} opened but can't read frames")
                        camera.release()
                        camera = None
                else:
                    print(f"❌ Failed to open camera {camera_index} with backend {backend}")
                    
            except Exception as e:
                print(f"❌ Error with camera {camera_index}: {e}")
                if camera:
                    camera.release()
                    camera = None
        
        if not camera or not camera.isOpened():
            raise Exception("No working camera found!")
            
        # Optimize camera settings
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FOURCC, 1196444237)  # MJPG format
        camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Capture image
        print("📸 Capturing image...")
        ret, frame = camera.read()
        if not ret or frame is None:
            raise Exception("Failed to capture image from camera")
            
        # Convert to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame, frame_rgb
        
    except Exception as e:
        print(f"❌ Camera error: {e}")
        raise e
    finally:
        if camera:
            camera.release()

@app.get("/")
async def root():
    return {"message": "✅ CV Detection API is running with smart confidence boosting"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detect/full-batch")
async def detect_batch(files: List[UploadFile] = File(...)):
    """
    Process multiple images with smart confidence boosting
    """
    results = []
    for f in files:
        content = await f.read()
        image = read_image(content)
        result = process_image(image, detection_mode="both")
        result["filename"] = f.filename
        results.append(result)
    return results

@app.post("/detect/single")
async def detect_single(file: UploadFile = File(...)):
    """
    Process single image with smart confidence boosting (detects both ingredients and nutrition)
    """
    content = await file.read()
    image = read_image(content)
    result = process_image(image, detection_mode="both")
    result["filename"] = file.filename
    return result

@app.post("/detect/custom")
async def detect_custom(
    file: UploadFile = File(...),
    detection_mode: str = Query("both", description="Detection mode: both, ingredients, nutrition"),
    preferred_language: Optional[str] = Query(None, description="Preferred language: en, es, fr, de, it")
):
    """
    Custom detection with mode and language selection
    - detection_mode: "both", "ingredients", or "nutrition"
    - preferred_language: "en", "es", "fr", "de", "it"
    """
    content = await file.read()
    image = read_image(content)
    result = process_image(image, preferred_language=preferred_language, detection_mode=detection_mode)
    result["filename"] = file.filename
    return result

@app.post("/camera/capture")
async def camera_capture():
    """
    📸 Capture image from camera and return it as base64
    """
    try:
        print("📸 Capturing image from camera...")
        frame, frame_rgb = capture_camera_image()
        
        # Encode image to base64
        import base64
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            image_bytes = buffer.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "success": True,
                "message": "✅ Image captured successfully!",
                "image_base64": image_base64,
                "size": f"{frame.shape[1]}x{frame.shape[0]}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to encode image")
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "❌ Failed to capture image"
        }

@app.post("/camera/capture-and-detect")
async def camera_capture_and_detect(
    detection_mode: str = Query("both", description="Detection mode: both, ingredients, nutrition"),
    preferred_language: Optional[str] = Query(None, description="Preferred language: en, es, fr, de, it")
):
    """
    📸 Capture image from camera and run detection immediately
    """
    try:
        print("📸 Capturing image from camera...")
        frame, frame_rgb = capture_camera_image()
        
        # Run detection
        print("🔍 Running detection...")
        start_time = time.time()
        result = process_image(frame_rgb, preferred_language=preferred_language, detection_mode=detection_mode)
        detection_time = time.time() - start_time
        
        # Encode image to base64
        import base64
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ret:
            image_bytes = buffer.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return {
                "success": True,
                "message": "✅ Capture and detection completed successfully!",
                "image_base64": image_base64,
                "detection_results": result,
                "processing_time": f"{detection_time:.2f}s",
                "size": f"{frame.shape[1]}x{frame.shape[0]}",
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise Exception("Failed to encode image")
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "❌ Failed to capture and detect"
        }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
