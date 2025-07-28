from detector import YoloDetector
from postprocess import boost_confidence, filter_by_label, boost_confidence_enhanced, smart_confidence_boost
from utils import read_image, draw_boxes, encode_base64
from paddleocr import PaddleOCR
from typing import List, Dict, Any
import re

# Try to import langdetect, but provide fallback if not available
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not available. Using fallback language detection.")

detector = YoloDetector()

# Initialize OCR models for different languages
ocr_models = {
    'en': PaddleOCR(use_angle_cls=True, lang='en'),
    'fr': PaddleOCR(use_angle_cls=True, lang='fr'),
    'es': PaddleOCR(use_angle_cls=True, lang='es'),
    'de': PaddleOCR(use_angle_cls=True, lang='german'),
    'it': PaddleOCR(use_angle_cls=True, lang='it')
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
    Run OCR with multi-language support
    """
    print(f"🔍 Starting OCR with preferred_lang={preferred_lang}")
    print(f"📦 Input image shape: {image.shape}")
    
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
        return final_result
    
    print(f"❌ No OCR results found")
    return {
        'text': '',
        'confidence': 0.0,
        'language': 'en',
        'ocr_model_used': 'none'
    }

def process_image(image, preferred_language: str = None, detection_mode: str = "both"):
    """
    Process image with multi-language OCR support and smart confidence boosting
    
    Args:
        image: Input image
        preferred_language: Preferred language for OCR ('en', 'es', 'fr', 'de', 'it')
        detection_mode: Detection mode ('both', 'ingredients', 'nutrition')
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

    # Apply smart confidence boosting with +/- logic using OCR text from each box
    if ingr_boxes or nutr_boxes:
        ingr_boxes, nutr_boxes = smart_confidence_boost(ingr_boxes, nutr_boxes)

    vis = draw_boxes(image, ingr_boxes + nutr_boxes)

    return {
        "ingredient_blocks": ingr_boxes,
        "nutrition_tables": nutr_boxes,
        "annotated_image": encode_base64(vis),
        "detection_mode": detection_mode
    }
