import base64
import cv2
import numpy as np
from typing import Union, List, Dict

def read_image(image_path: Union[str, bytes, np.ndarray]) -> np.ndarray:
    """
    Read image from various sources (file path, bytes, or numpy array)
    """
    if isinstance(image_path, str):
        # Read from file path
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_path}")
    elif isinstance(image_path, bytes):
        # Read from bytes
        nparr = np.frombuffer(image_path, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes")
    elif isinstance(image_path, np.ndarray):
        # Already a numpy array
        image = image_path.copy()
    else:
        raise ValueError(f"Unsupported image type: {type(image_path)}")
    
    return image

def draw_boxes(image, boxes):
    for b in boxes:
        x1, y1, x2, y2 = b["box"]
        label = f'{b["label"]} ({b["confidence"]})'
        cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return image

def encode_base64(image):
    _, buf = cv2.imencode('.jpg', image)
    return base64.b64encode(buf).decode('utf-8')

def resize_image(image: np.ndarray, target_size: tuple = (640, 640)) -> np.ndarray:
    """
    Resize image to target size while maintaining aspect ratio
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scaling factor
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded

def crop_image(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """
    Crop image using bounding box coordinates
    """
    # Ensure coordinates are within image bounds
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    
    return image[y1:y2, x1:x2]
