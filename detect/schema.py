from pydantic import BaseModel
from typing import List, Tuple

class DetectionBox(BaseModel):
    label: str
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float

class DetectionResponse(BaseModel):
    results: List[DetectionBox]
