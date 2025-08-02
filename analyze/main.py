from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import time
import json

from .analyzer import analyze_ingredients, summarize_nutrition, analyze_food_package_comprehensive
from .llm_prompter import Language, llm_manager

app = FastAPI(
    title="CV Analysis API",
    description="AI-powered food analysis with LangChain integration and multilingual support (English/French)",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class AnalysisRequest(BaseModel):
    text: str
    language: str = "en"  # "en" or "fr"
    analysis_type: Optional[str] = "auto"  # "ingredients", "nutrition", "comprehensive", "auto"

class AnalysisResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    language: str
    model_used: Optional[str] = None
    processing_time: Optional[float] = None
    confidence: Optional[float] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_available: List[str]
    language_support: List[str]
    version: str

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "✅ CV Analysis API is running with LangChain integration",
        "version": "3.0.0",
        "features": [
            "LangChain LLM integration",
            "Multilingual support (English/French)",
            "Enhanced error handling",
            "Multiple LLM providers (OpenAI, Anthropic, Ollama)",
            "Smart fallback mechanisms"
        ],
        "endpoints": {
            "/health": "Service health check",
            "/analyze/ingredients": "Analyze ingredient lists",
            "/analyze/nutrition": "Analyze nutrition facts",
            "/analyze/comprehensive": "Comprehensive food analysis",
            "/analyze/auto": "Auto-detect and analyze content"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Enhanced health check with model availability"""
    models_available = list(llm_manager.models.keys())
    language_support = ["en", "fr"]
    
    return HealthResponse(
        status="healthy",
        models_available=models_available,
        language_support=language_support,
        version="3.0.0"
    )

@app.post("/analyze/ingredients", response_model=AnalysisResponse)
async def analyze_ingredients_endpoint(request: AnalysisRequest):
    """Analyze ingredient lists with multilingual support"""
    start_time = time.time()
    
    try:
        # Validate language
        if request.language not in ["en", "fr"]:
            raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
        
        language = Language.FRENCH if request.language == "fr" else Language.ENGLISH
        
        # Perform analysis
        result = analyze_ingredients(request.text, language)
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            data=result,
            language=request.language,
            processing_time=processing_time,
            confidence=0.9
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return AnalysisResponse(
            success=False,
            data={},
            language=request.language,
            processing_time=processing_time,
            error=str(e)
        )

@app.post("/analyze/nutrition", response_model=AnalysisResponse)
async def analyze_nutrition_endpoint(request: AnalysisRequest):
    """Analyze nutrition facts with multilingual support"""
    start_time = time.time()
    
    try:
        # Validate language
        if request.language not in ["en", "fr"]:
            raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
        
        language = Language.FRENCH if request.language == "fr" else Language.ENGLISH
        
        # Perform analysis
        result = summarize_nutrition(request.text, language)
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            data=result,
            language=request.language,
            processing_time=processing_time,
            confidence=0.9
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return AnalysisResponse(
            success=False,
            data={},
            language=request.language,
            processing_time=processing_time,
            error=str(e)
        )

@app.post("/analyze/comprehensive", response_model=AnalysisResponse)
async def analyze_comprehensive_endpoint(request: AnalysisRequest):
    """Comprehensive food analysis with multilingual support"""
    start_time = time.time()
    
    try:
        # Validate language
        if request.language not in ["en", "fr"]:
            raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
        
        language = Language.FRENCH if request.language == "fr" else Language.ENGLISH
        
        # Perform comprehensive analysis
        result = analyze_food_package_comprehensive(request.text, "comprehensive")
        
        # Add language information to result
        result["analysis_language"] = request.language
        result["multilingual_support"] = True
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            data=result,
            language=request.language,
            processing_time=processing_time,
            confidence=0.9
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return AnalysisResponse(
            success=False,
            data={},
            language=request.language,
            processing_time=processing_time,
            error=str(e)
        )

@app.post("/analyze/auto", response_model=AnalysisResponse)
async def analyze_auto_endpoint(request: AnalysisRequest):
    """Auto-detect content type and analyze with multilingual support"""
    start_time = time.time()
    
    try:
        # Validate language
        if request.language not in ["en", "fr"]:
            raise HTTPException(status_code=400, detail="Language must be 'en' or 'fr'")
        
        language = Language.FRENCH if request.language == "fr" else Language.ENGLISH
        
        # Auto-detect content type and analyze
        text_lower = request.text.lower()
        
        if any(keyword in text_lower for keyword in ["ingredient", "ingrédient", "composant"]):
            result = analyze_ingredients(request.text, language)
            analysis_type = "ingredients"
        elif any(keyword in text_lower for keyword in ["nutrition", "calorie", "vitamin", "vitamine", "protein", "protéine"]):
            result = summarize_nutrition(request.text, language)
            analysis_type = "nutrition"
        else:
            # Default to comprehensive analysis
            result = analyze_food_package_comprehensive(request.text, "auto")
            analysis_type = "comprehensive"
        
        # Add metadata
        result["analysis_type"] = analysis_type
        result["language"] = request.language
        result["auto_detected"] = True
        
        processing_time = time.time() - start_time
        
        return AnalysisResponse(
            success=True,
            data=result,
            language=request.language,
            processing_time=processing_time,
            confidence=0.85
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        return AnalysisResponse(
            success=False,
            data={},
            language=request.language,
            processing_time=processing_time,
            error=str(e)
        )

@app.get("/models/available")
async def get_available_models():
    """Get list of available LLM models"""
    return {
        "models": list(llm_manager.models.keys()),
        "default_model": "ollama" if "ollama" in llm_manager.models else list(llm_manager.models.keys())[0] if llm_manager.models else None,
        "total_models": len(llm_manager.models)
    }

@app.get("/languages/supported")
async def get_supported_languages():
    """Get list of supported languages"""
    return {
        "languages": [
            {"code": "en", "name": "English", "flag": "🇺🇸"},
            {"code": "fr", "name": "Français", "flag": "🇫🇷"}
        ],
        "default": "en"
    }

@app.post("/translate")
async def translate_text(text: str, target_language: str = "fr"):
    """Translate text to target language"""
    try:
        translated = llm_manager.translate_text(text, target_language)
        return {
            "success": True,
            "original": text,
            "translated": translated,
            "target_language": target_language
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "original": text,
            "target_language": target_language
        }

# Example usage endpoint
@app.get("/example")
async def get_example():
    """Get example analysis requests"""
    return {
        "examples": {
            "ingredients_en": {
                "text": "Sugar, Salt, Milk, Wheat Flour, Artificial Flavors, Preservatives",
                "language": "en",
                "analysis_type": "ingredients"
            },
            "ingredients_fr": {
                "text": "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels, Conservateurs",
                "language": "fr",
                "analysis_type": "ingredients"
            },
            "nutrition_en": {
                "text": "Energy: 480kcal, Protein: 8g, Fat: 12g, Carbohydrates: 45g, Sodium: 220mg",
                "language": "en",
                "analysis_type": "nutrition"
            },
            "nutrition_fr": {
                "text": "Énergie: 480kcal, Protéines: 8g, Lipides: 12g, Glucides: 45g, Sodium: 220mg",
                "language": "fr",
                "analysis_type": "nutrition"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 