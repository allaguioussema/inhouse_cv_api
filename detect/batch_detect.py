from detector import YoloDetector
from postprocess import boost_confidence, filter_by_label, boost_confidence_enhanced, smart_confidence_boost
from utils import read_image, draw_boxes, encode_base64
from paddleocr import PaddleOCR
from typing import List, Dict, Any
import re

# Import analyze module for LLM analysis with proper error handling
import sys
import os

# Add the parent directory to the path to import analyze module
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Try to import analyze module with fallback
try:
    from analyze.analyzer import analyze_ingredients, summarize_nutrition, analyze_food_package_comprehensive, detect_food_content_type, detect_food_content_type_fallback
    from analyze.llm_prompter import prompt_llm, Language
    ANALYZE_MODULE_AVAILABLE = True
    print("✅ Analyze module imported successfully")
except ImportError as e:
    print(f"⚠️ Analyze module not available: {e}")
    ANALYZE_MODULE_AVAILABLE = False
    # Create fallback functions
    def analyze_ingredients(text: str, language=None):
        return {"error": "Analyze module not available", "fallback": True}
    
    def summarize_nutrition(text: str, language=None):
        return {"error": "Analyze module not available", "fallback": True}
    
    def analyze_food_package_comprehensive(text: str, detection_type: str):
        return {"error": "Analyze module not available", "fallback": True}
    
    def detect_food_content_type(text: str):
        return {"content_type": "unknown", "confidence": 0.0, "fallback": True}
    
    def detect_food_content_type_fallback(text: str):
        return {"content_type": "unknown", "confidence": 0.0, "fallback": True}
    
    def prompt_llm(prompt: str, language=None):
        return "Analyze module not available"
    
    class Language:
        ENGLISH = "en"
        FRENCH = "fr"

# Try to import langdetect, but provide fallback if not available
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Using fallback language detection.")

detector = YoloDetector()

# Initialize OCR models for different languages with performance optimizations
ocr_models = {
    'en': PaddleOCR(use_angle_cls=True, lang='en'),
    'fr': PaddleOCR(use_angle_cls=True, lang='fr'),
    'es': PaddleOCR(use_angle_cls=True, lang='es'),
    'de': PaddleOCR(use_angle_cls=True, lang='german'),
    'it': PaddleOCR(use_angle_cls=True, lang='it')
}

# Cache for OCR results to avoid re-processing same images
ocr_cache = {}

def clear_ocr_cache():
    """
    Clear the OCR cache to free memory
    """
    global ocr_cache
    ocr_cache.clear()
    print("🧹 OCR cache cleared")

def get_ocr_cache_stats():
    """
    Get statistics about the OCR cache
    """
    return {
        'cache_size': len(ocr_cache),
        'cache_keys': list(ocr_cache.keys())[:10]  # First 10 keys for debugging
    }

def detect_language_simple(text: str) -> str:
    """
    Simple language detection using common words and patterns
    """
    if not text.strip():
        return 'en'
    
    text_lower = text.lower()
    
    # Spanish patterns (more comprehensive)
    spanish_words = ['calorías', 'valor', 'ingredientes', 'porción', 'energía', 'proteínas', 'carbohidratos', 'grasas', 'ingredientes', 'producto', 'contiene', 'elaborado', 'equipo', 'procesa', 'trigo', 'soya', 'leche', 'huevo', 'nueces', 'cacahuate', 'coco']
    if any(word in text_lower for word in spanish_words):
        return 'es'
    
    # French patterns
    french_words = ['calories', 'valeur', 'ingrédients', 'portion', 'énergie', 'protéines', 'glucides', 'lipides']
    if any(word in text_lower for word in french_words):
        return 'fr'
    
    # German patterns
    german_words = ['kalorien', 'wert', 'zutaten', 'portion', 'energie', 'proteine', 'kohlenhydrate', 'fette']
    if any(word in text_lower for word in german_words):
        return 'de'
    
    # Italian patterns
    italian_words = ['calorie', 'valore', 'ingredienti', 'porzione', 'energia', 'proteine', 'carboidrati', 'grassi']
    if any(word in text_lower for word in italian_words):
        return 'it'
    
    # Default to English
    return 'en'

def detect_language(text: str) -> str:
    """
    Detect the language of the text
    """
    # Always try simple detection first for food packaging
    simple_lang = detect_language_simple(text)
    if simple_lang != 'en':  # If simple detection found a specific language, use it
        return simple_lang
    
    # Fallback to langdetect only if simple detection didn't find anything
    if LANGDETECT_AVAILABLE:
        try:
            if not text.strip():
                return 'en'  # Default to English if no text
            lang = langdetect.detect(text)
            # Map detected language to supported OCR languages
            lang_mapping = {
                'en': 'en', 'fr': 'fr', 'es': 'es', 'de': 'de', 
                'it': 'it', 'ca': 'es', 'nl': 'en'
            }
            return lang_mapping.get(lang, 'en')  # Default to English instead of 'multi'
        except:
            return 'en'  # Default to English on error
    else:
        return 'en'  # Default to English if langdetect not available

def run_ocr(image, preferred_lang: str = None) -> Dict[str, Any]:
    """
    Enhanced OCR with multi-language support and performance optimizations
    """
    print(f"🔍 Starting enhanced OCR with preferred_lang={preferred_lang}")
    print(f"📦 Input image shape: {image.shape}")
    
    # Create a simple hash for caching (basic implementation)
    import hashlib
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    cache_key = f"{image_hash}_{preferred_lang}"
    
    # Check cache first
    if cache_key in ocr_cache:
        print(f"✅ Using cached OCR result for {preferred_lang}")
        return ocr_cache[cache_key]
    
    results = {}
    
    # Try preferred language first if specified
    if preferred_lang and preferred_lang in ocr_models:
        try:
            print(f"🔍 Trying {preferred_lang} OCR model...")
            result = ocr_models[preferred_lang].ocr(image)
            print(f"📝 Raw OCR result: {result}")
            
            if result and len(result) > 0:
                # Handle new PaddleOCR result structure
                ocr_result = result[0]  # Get the OCRResult object
                print(f"📝 OCR result structure: {ocr_result}")
                
                if 'rec_texts' in ocr_result and ocr_result['rec_texts']:
                    text = " ".join(ocr_result['rec_texts'])
                    confidence = sum(ocr_result['rec_scores']) / len(ocr_result['rec_scores']) if ocr_result['rec_scores'] else 0.8
                    
                    # Enhanced text cleaning
                    text = clean_ocr_text(text)
                    
                    results[preferred_lang] = {
                        'text': text,
                        'confidence': confidence
                    }
                    print(f"✅ {preferred_lang} OCR successful: '{text[:50]}...'")
                else:
                    print(f"❌ {preferred_lang} OCR: No text found in result")
            else:
                print(f"❌ {preferred_lang} OCR: No result returned")
        except Exception as e:
            print(f"❌ Error with {preferred_lang} OCR: {e}")
    
    # If no results yet, try English first
    if not results and 'en' in ocr_models:
        try:
            print(f"🔍 Trying English OCR model...")
            result = ocr_models['en'].ocr(image)
            print(f"📝 Raw English OCR result: {result}")
            
            if result and len(result) > 0:
                # Handle new PaddleOCR result structure
                ocr_result = result[0]  # Get the OCRResult object
                print(f"📝 English OCR result structure: {ocr_result}")
                
                if 'rec_texts' in ocr_result and ocr_result['rec_texts']:
                    text = " ".join(ocr_result['rec_texts'])
                    confidence = sum(ocr_result['rec_scores']) / len(ocr_result['rec_scores']) if ocr_result['rec_scores'] else 0.8
                    
                    # Enhanced text cleaning
                    text = clean_ocr_text(text)
                    
                    results['en'] = {
                        'text': text,
                        'confidence': confidence
                    }
                    print(f"✅ English OCR successful: '{text[:50]}...'")
                else:
                    print(f"❌ English OCR: No text found in result")
            else:
                print(f"❌ English OCR: No result returned")
        except Exception as e:
            print(f"❌ Error with English OCR: {e}")
    
    # Try other available models as fallback
    if not results:
        for lang, model in ocr_models.items():
            if lang != 'en' and lang not in results:  # Skip English since we already tried it
                try:
                    print(f"🔍 Trying {lang} OCR model as fallback...")
                    result = model.ocr(image)
                    print(f"📝 Raw {lang} OCR result: {result}")
                    
                    if result and len(result) > 0:
                        # Handle new PaddleOCR result structure
                        ocr_result = result[0]  # Get the OCRResult object
                        print(f"📝 {lang} OCR result structure: {ocr_result}")
                        
                        if 'rec_texts' in ocr_result and ocr_result['rec_texts']:
                            text = " ".join(ocr_result['rec_texts'])
                            confidence = sum(ocr_result['rec_scores']) / len(ocr_result['rec_scores']) if ocr_result['rec_scores'] else 0.8
                            
                            # Enhanced text cleaning
                            text = clean_ocr_text(text)
                            
                            results[lang] = {
                                'text': text,
                                'confidence': confidence
                            }
                            print(f"✅ {lang} OCR successful: '{text[:50]}...'")
                            break
                        else:
                            print(f"❌ {lang} OCR: No text found in result")
                    else:
                        print(f"❌ {lang} OCR: No result returned")
                except Exception as e:
                    print(f"❌ Error with {lang} OCR: {e}")
                    continue
    
    # Return the best result (highest confidence)
    if results:
        best_lang = max(results.keys(), key=lambda k: results[k]['confidence'])
        detected_lang = detect_language(results[best_lang]['text'])
        
        final_result = {
            'text': results[best_lang]['text'],
            'confidence': results[best_lang]['confidence'],
            'language': detected_lang,
            'ocr_model_used': best_lang
        }
        print(f"✅ Final OCR result: {final_result}")
        
        # Cache the result
        ocr_cache[cache_key] = final_result
        
        return final_result
    
    print(f"❌ No OCR results found")
    empty_result = {
        'text': '',
        'confidence': 0.0,
        'language': 'en',
        'ocr_model_used': 'none'
    }
    
    # Cache the empty result too
    ocr_cache[cache_key] = empty_result
    
    return empty_result

def clean_ocr_text(text: str) -> str:
    """
    Clean and enhance OCR text for better keyword matching
    """
    if not text:
        return ""
    
    # Remove common OCR artifacts
    text = text.replace('|', 'I')  # Common OCR mistake
    text = text.replace('0', 'O')  # Common OCR mistake in certain contexts
    text = text.replace('1', 'I')  # Common OCR mistake
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Fix common OCR errors in nutrition/ingredient context
    nutrition_fixes = {
        'calorles': 'calories',
        'proteln': 'protein',
        'carbohydrate': 'carbohydrate',
        'fat': 'fat',
        'sodlum': 'sodium',
        'fiber': 'fiber',
        'sugar': 'sugar',
        'cholesterol': 'cholesterol',
        'vitamin': 'vitamin',
        'mineral': 'mineral'
    }
    
    for wrong, correct in nutrition_fixes.items():
        text = text.replace(wrong, correct)
    
    return text

def analyze_ocr_with_llm(ocr_text: str, detection_type: str, use_llm: bool = True, preferred_language: str = None) -> Dict[str, Any]:
    """
    Enhanced OCR analysis using smart LLM with content type detection and comprehensive analysis
    
    Args:
        ocr_text: Text extracted from OCR
        detection_type: "ingredient" or "nutrition"
        use_llm: Whether to use LLM analysis (can be disabled for performance)
        preferred_language: Preferred language for analysis (en, fr, etc.)
    
    Returns:
        Dictionary with comprehensive analysis results
    """
    if not use_llm or not ocr_text.strip():
        return {
            "llm_analysis": False,
            "reason": "LLM disabled or no text to analyze"
        }
    
    # Check if analyze module is available
    if not ANALYZE_MODULE_AVAILABLE:
        return {
            "llm_analysis": False,
            "reason": "Analyze module not available",
            "fallback_analysis": extract_simple_keywords(ocr_text)
        }
    
    try:
        print(f"🤖 Starting enhanced LLM analysis for {detection_type} text...")
        
        # Detect language if not provided
        if not preferred_language:
            detected_lang = detect_language(ocr_text)
            preferred_language = detected_lang
            print(f"🌍 Detected language: {preferred_language}")
        
        # Set language for analysis
        if preferred_language == "fr":
            language = Language.FRENCH
        else:
            language = Language.ENGLISH
        
        # First, detect the actual content type of the text
        try:
            content_type_result = detect_food_content_type(ocr_text)
            print(f"🔍 Content type detection: {content_type_result.get('content_type', 'unknown')} (confidence: {content_type_result.get('confidence', 0):.2f})")
        except Exception as e:
            print(f"⚠️ LLM content type detection failed, using fallback: {e}")
            content_type_result = detect_food_content_type_fallback(ocr_text)
            print(f"🔍 Fallback content type detection: {content_type_result.get('content_type', 'unknown')} (confidence: {content_type_result.get('confidence', 0):.2f})")
        
        # Use comprehensive analysis that adapts to content type
        if detection_type == "ingredient":
            # Analyze ingredients with enhanced capabilities
            analysis_result = analyze_ingredients(ocr_text, language)
            analysis_result["analysis_type"] = "enhanced_ingredient_analysis"
            
        elif detection_type == "nutrition":
            # Analyze nutrition with enhanced capabilities
            analysis_result = summarize_nutrition(ocr_text, language)
            analysis_result["analysis_type"] = "enhanced_nutrition_analysis"
            
        else:
            # Use comprehensive analysis that adapts to content
            analysis_result = analyze_food_package_comprehensive(ocr_text, detection_type)
            analysis_result["analysis_type"] = "comprehensive_analysis"
        
        # Add metadata and content type information
        analysis_result["llm_analysis"] = True
        analysis_result["text_length"] = len(ocr_text)
        analysis_result["detection_type"] = detection_type
        analysis_result["content_type_detection"] = content_type_result
        analysis_result["preferred_language"] = preferred_language
        analysis_result["analyze_module_available"] = True
        
        # Add confidence scoring based on content type match
        if content_type_result.get('content_type') == detection_type:
            analysis_result["content_type_match"] = True
            analysis_result["content_type_confidence"] = content_type_result.get('confidence', 0)
        else:
            analysis_result["content_type_match"] = False
            analysis_result["content_type_confidence"] = content_type_result.get('confidence', 0)
            print(f"⚠️  Content type mismatch: Detected as {detection_type} but LLM classified as {content_type_result.get('content_type')}")
        
        print(f"✅ Enhanced LLM analysis completed for {detection_type} in {preferred_language}")
        return analysis_result
        
    except Exception as e:
        print(f"❌ Enhanced LLM analysis failed: {e}")
        # Use fallback analysis
        fallback_content = detect_food_content_type_fallback(ocr_text)
        fallback_keywords = extract_simple_keywords(ocr_text)
        
        return {
            "llm_analysis": False,
            "error": str(e),
            "detection_type": detection_type,
            "preferred_language": preferred_language,
            "analyze_module_available": ANALYZE_MODULE_AVAILABLE,
            "fallback_analysis": {
                "text_length": len(ocr_text),
                "simple_keywords": fallback_keywords,
                "content_type_detection": fallback_content
            },
            "content_type_detection": fallback_content,
            "analysis_type": "fallback_analysis"
        }

def extract_simple_keywords(text: str) -> Dict[str, Any]:
    """
    Fallback keyword extraction when LLM analysis fails
    """
    text_lower = text.lower()
    
    # Simple keyword detection
    allergens = ["milk", "wheat", "soy", "egg", "nuts", "gluten", "peanut", "almond", "walnut", "sesame"]
    additives = ["preservative", "color", "flavor", "stabilizer", "emulsifier", "xanthan", "gum"]
    nutrition = ["calories", "protein", "fat", "carbohydrate", "sugar", "salt", "fiber"]
    
    found_allergens = [word for word in allergens if word in text_lower]
    found_additives = [word for word in additives if word in text_lower]
    found_nutrition = [word for word in nutrition if word in text_lower]
    
    return {
        "allergens": found_allergens,
        "additives": found_additives,
        "nutrition_keywords": found_nutrition,
        "analysis_method": "simple_keyword_extraction"
    }

def process_image(image, preferred_language: str = None, detection_mode: str = "both", use_llm: bool = True):
    """
    Process image with multi-language OCR support, smart confidence boosting, and LLM analysis
    
    Args:
        image: Input image
        preferred_language: Preferred language for OCR ('en', 'es', 'fr', 'de', 'it')
        detection_mode: Detection mode ('both', 'ingredients', 'nutrition')
        use_llm: Whether to use LLM analysis after OCR (default: True)
    """
    # Run detection based on mode
    if detection_mode == "ingredients":
        ingr_boxes = detector.predict(image, "ingredient")
        nutr_boxes = []
    elif detection_mode == "nutrition":
        ingr_boxes = []
        nutr_boxes = detector.predict(image, "nutrition")
    else:  # "both"
        ingr_boxes = detector.predict(image, "ingredient")
        nutr_boxes = detector.predict(image, "nutrition")

    # Fix: Use actual labels from models
    ingr_boxes = filter_by_label(ingr_boxes, "Ingredients")
    nutr_boxes = filter_by_label(nutr_boxes, "NUTRITION-TABLE")

    # Run OCR on each detected bounding box
    from utils import crop_image
    
    print(f"🔍 Found {len(ingr_boxes)} ingredient boxes and {len(nutr_boxes)} nutrition boxes")
    
    # Process ingredient boxes
    for i, box in enumerate(ingr_boxes):
        # Crop the region from the image
        x1, y1, x2, y2 = int(box['box'][0]), int(box['box'][1]), int(box['box'][2]), int(box['box'][3])
        print(f"🔍 Processing ingredient box {i}: bbox=[{x1},{y1},{x2},{y2}]")
        
        cropped_region = crop_image(image, x1, y1, x2, y2)
        print(f"📦 Cropped region shape: {cropped_region.shape}")
        
        # Check if cropped region is valid
        if cropped_region.size == 0:
            print(f"❌ Empty cropped region for ingredient box {i}")
            box['ocr_text'] = ""
            box['ocr_confidence'] = 0.0
            box['ocr_language'] = 'en'
            continue
        
        # Run OCR on this specific region
        ocr_result = run_ocr(cropped_region, preferred_language)
        print(f"📝 OCR result for ingredient box {i}: {ocr_result}")
        
        box['ocr_text'] = ocr_result['text']
        box['ocr_confidence'] = ocr_result['confidence']
        box['ocr_language'] = ocr_result['language']
        print(f"🔍 Ingredient box {i} OCR: {ocr_result['language']} - '{ocr_result['text'][:50]}...'")
        
        # Perform LLM analysis on ingredient text
        if use_llm and ocr_result['text'].strip():
            llm_analysis = analyze_ocr_with_llm(ocr_result['text'], "ingredient", use_llm, preferred_language)
            box['llm_analysis'] = llm_analysis
            print(f"🤖 LLM analysis for ingredient box {i}: {llm_analysis.get('analysis_type', 'unknown')}")

    # Process nutrition boxes
    for i, box in enumerate(nutr_boxes):
        # Crop the region from the image
        x1, y1, x2, y2 = int(box['box'][0]), int(box['box'][1]), int(box['box'][2]), int(box['box'][3])
        print(f"🔍 Processing nutrition box {i}: bbox=[{x1},{y1},{x2},{y2}]")
        
        cropped_region = crop_image(image, x1, y1, x2, y2)
        print(f"📦 Cropped region shape: {cropped_region.shape}")
        
        # Check if cropped region is valid
        if cropped_region.size == 0:
            print(f"❌ Empty cropped region for nutrition box {i}")
            box['ocr_text'] = ""
            box['ocr_confidence'] = 0.0
            box['ocr_language'] = 'en'
            continue
        
        # Run OCR on this specific region
        ocr_result = run_ocr(cropped_region, preferred_language)
        print(f"📝 OCR result for nutrition box {i}: {ocr_result}")
        
        box['ocr_text'] = ocr_result['text']
        box['ocr_confidence'] = ocr_result['confidence']
        box['ocr_language'] = ocr_result['language']
        print(f"🔍 Nutrition box {i} OCR: {ocr_result['language']} - '{ocr_result['text'][:50]}...'")
        
        # Perform LLM analysis on nutrition text
        if use_llm and ocr_result['text'].strip():
            llm_analysis = analyze_ocr_with_llm(ocr_result['text'], "nutrition", use_llm, preferred_language)
            box['llm_analysis'] = llm_analysis
            print(f"🤖 LLM analysis for nutrition box {i}: {llm_analysis.get('analysis_type', 'unknown')}")

    # Apply smart confidence boosting with +/- logic using OCR text from each box
    if ingr_boxes or nutr_boxes:
        ingr_boxes, nutr_boxes = smart_confidence_boost(ingr_boxes, nutr_boxes)

    vis = draw_boxes(image, ingr_boxes + nutr_boxes)

    return {
        "ingredient_blocks": ingr_boxes,
        "nutrition_tables": nutr_boxes,
        "annotated_image": encode_base64(vis),
        "detection_mode": detection_mode,
        "llm_analysis_enabled": use_llm
    }
