import os
from ultralytics import YOLO
import torch

class YoloDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}

        model_paths = {
            "nutrition": "models/best NUT.pt",
            "ingredient": "models/best ING.pt"
        }

        for key, path in model_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            self.models[key] = YOLO(path).to(self.device)

    def predict(self, image, model_type: str):
        if model_type not in self.models:
            raise ValueError(f"Invalid model type '{model_type}'. Choose from: {list(self.models.keys())}")

        model = self.models[model_type]
        results = model(image)[0]  # Assuming single image input

        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            label_idx = int(box.cls[0])
            label = model.names[label_idx]
            detections.append({
                "label": label,
                "box": [x1, y1, x2, y2],
                "confidence": round(conf, 3)
            })

        return detections
