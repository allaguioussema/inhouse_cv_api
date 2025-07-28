from fastapi import FastAPI, UploadFile, File, Query
from typing import List, Optional
from batch_detect import process_image, read_image
import uvicorn

app = FastAPI(
    title="CV Detection API",
    description="Object detection and OCR for food packaging (ingredients + nutrition) with smart confidence boosting",
    version="2.0.0"
)

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

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
