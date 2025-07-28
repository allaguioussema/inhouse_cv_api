def filter_by_label(boxes, expected_label):
    return [b for b in boxes if b["label"] == expected_label]

def boost_confidence(boxes, text, keywords):
    for b in boxes:
        if any(k in text.lower() for k in keywords):
            b["confidence"] = round(min(b["confidence"] + 0.1, 1.0), 3)
    return boxes

def boost_confidence_enhanced(boxes, text, keywords, ocr_confidence=0.0, detection_type="unknown"):
    """
    Enhanced confidence boosting with +/- logic
    """
    for b in boxes:
        # Original boost for keywords
        if any(k in text.lower() for k in keywords):
            b["confidence"] = round(min(b["confidence"] + 0.1, 1.0), 3)
        
        # Additional +1 boost conditions - only apply to relevant detections
        boost_conditions = []
        
        # Condition 1: High OCR confidence (>0.8)
        if ocr_confidence > 0.8:
            boost_conditions.append("High OCR confidence")
        
        # Condition 2: Text contains relevant keywords for this detection type
        if detection_type == "nutrition":
            nutrition_words = ["calories", "nutrition", "serving", "protein", "fat", "carbohydrate", "vitamin", "mineral"]
            if any(word in text.lower() for word in nutrition_words):
                boost_conditions.append("Nutrition keywords found")
        elif detection_type == "ingredient":
            ingredient_words = ["ingredients", "ingredientes", "milk", "wheat", "soy", "egg", "nuts", "gluten"]
            if any(word in text.lower() for word in ingredient_words):
                boost_conditions.append("Ingredient keywords found")
        
        # Condition 3: Text length is substantial (>50 chars)
        if len(text) > 50:
            boost_conditions.append("Substantial text length")
        
        # Apply +1 boost if any conditions are met
        if boost_conditions:
            b["confidence"] = round(min(b["confidence"] + 1.0, 1.0), 3)
            print(f"✅ Confidence boosted by +1 for {detection_type} detection - conditions: {', '.join(boost_conditions)}")
    
    return boxes

def smart_confidence_boost(ingr_boxes, nutr_boxes):
    """
    Smart confidence boosting with +/- logic using OCR text from individual boxes
    """
    # Check for ingredient keywords (multi-language)
    ingredient_keywords = [
        # English
        "ingredients", "milk", "wheat", "soy", "egg", "nuts", "gluten",
        # Spanish
        "ingredientes", "trigo", "soya", "leche", "huevo", "nueces"
    ]
    
    # Check for nutrition keywords (multi-language)
    nutrition_keywords = [
        # English
        "calories", "nutrition", "protein", "fat", "carbohydrate", "serving",
        # Spanish
        "calorías", "proteínas", "grasas", "carbohidratos", "porción"
    ]
    
    # Apply smart boosting to ingredient boxes
    for b in ingr_boxes:
        if 'ocr_text' in b:
            text_lower = b['ocr_text'].lower()
            has_ingredient_keywords = any(word in text_lower for word in ingredient_keywords)
            has_nutrition_keywords = any(word in text_lower for word in nutrition_keywords)
            
            if has_ingredient_keywords:
                b["confidence"] = round(min(b["confidence"] + 0.3, 1.0), 3)
                print(f"✅ Ingredient confidence +0.3 (ingredient keywords found in box)")
            if has_nutrition_keywords:
                b["confidence"] = round(max(b["confidence"] - 0.5, 0.0), 3)
                print(f"❌ Ingredient confidence -0.5 (nutrition keywords found in box)")
    
    # Apply smart boosting to nutrition boxes
    for b in nutr_boxes:
        if 'ocr_text' in b:
            text_lower = b['ocr_text'].lower()
            has_ingredient_keywords = any(word in text_lower for word in ingredient_keywords)
            has_nutrition_keywords = any(word in text_lower for word in nutrition_keywords)
            
            if has_nutrition_keywords:
                b["confidence"] = round(min(b["confidence"] + 0.3, 1.0), 3)
                print(f"✅ Nutrition confidence +0.3 (nutrition keywords found in box)")
            if has_ingredient_keywords:
                b["confidence"] = round(max(b["confidence"] - 0.5, 0.0), 3)
                print(f"❌ Nutrition confidence -0.5 (ingredient keywords found in box)")
    
    return ingr_boxes, nutr_boxes
