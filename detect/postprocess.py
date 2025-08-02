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
    Enhanced smart confidence boosting with sophisticated +/- logic using OCR text from individual boxes
    """
    # Enhanced ingredient keywords (multi-language)
    ingredient_keywords = [
        # English
        "ingredients", "milk", "wheat", "soy", "egg", "nuts", "gluten", "contains", "allergen", "may contain",
        "dairy", "lactose", "casein", "whey", "peanut", "almond", "walnut", "pecan", "hazelnut", "pistachio",
        "sesame", "shellfish", "fish", "crustacean", "mollusk", "sulfite", "sulfites", "sulfiting", "sulfiting agent",
        # Spanish
        "ingredientes", "trigo", "soya", "leche", "huevo", "nueces", "contiene", "alergeno", "puede contener",
        "lactosa", "caseina", "suero", "cacahuate", "almendra", "nuez", "pecana", "avellana", "pistacho",
        "sesamo", "marisco", "pescado", "crustaceo", "molusco", "sulfito", "sulfitos",
        # French
        "ingrédients", "lait", "blé", "soja", "œuf", "noix", "gluten", "contient", "allergène", "peut contenir",
        "lactose", "caséine", "lactosérum", "arachide", "amande", "noix", "noix de pécan", "noisette", "pistache",
        "sésame", "crustacé", "poisson", "mollusque", "sulfite", "sulfites",
        # German
        "zutaten", "milch", "weizen", "soja", "ei", "nüsse", "gluten", "enthält", "allergen", "kann enthalten",
        "laktose", "kasein", "molke", "erdnuss", "mandel", "walnuss", "pecannuss", "haselnuss", "pistazie",
        "sesam", "schalentier", "fisch", "weichtier", "sulfit", "sulfite",
        # Italian
        "ingredienti", "latte", "frumento", "soia", "uovo", "noci", "glutine", "contiene", "allergene", "può contenire",
        "lattosio", "caseina", "siero", "arachide", "mandorla", "noce", "noce pecan", "nocciola", "pistacchio",
        "sesamo", "crostaceo", "pesce", "mollusco", "solfito", "solfiti"
    ]
    
    # Enhanced nutrition keywords (multi-language)
    nutrition_keywords = [
        # English
        "calories", "nutrition", "protein", "fat", "carbohydrate", "serving", "nutrition facts", "per serving",
        "total fat", "saturated fat", "trans fat", "cholesterol", "sodium", "total carbohydrate", "dietary fiber",
        "sugars", "added sugars", "vitamin", "mineral", "calcium", "iron", "potassium", "vitamin d", "vitamin c",
        "thiamin", "riboflavin", "niacin", "vitamin b6", "vitamin b12", "folate", "biotin", "pantothenic acid",
        # Spanish
        "calorías", "proteínas", "grasas", "carbohidratos", "porción", "información nutricional", "por porción",
        "grasa total", "grasa saturada", "grasa trans", "colesterol", "sodio", "carbohidratos totales", "fibra dietética",
        "azúcares", "azúcares añadidos", "vitamina", "mineral", "calcio", "hierro", "potasio", "vitamina d", "vitamina c",
        # French
        "calories", "protéines", "lipides", "glucides", "portion", "valeur nutritive", "par portion",
        "lipides totaux", "lipides saturés", "lipides trans", "cholestérol", "sodium", "glucides totaux", "fibres alimentaires",
        "sucres", "sucres ajoutés", "vitamine", "minéral", "calcium", "fer", "potassium", "vitamine d", "vitamine c",
        # German
        "kalorien", "proteine", "fette", "kohlenhydrate", "portion", "nährwertangaben", "pro portion",
        "gesamtfett", "gesättigte fettsäuren", "trans-fettsäuren", "cholesterin", "natrium", "gesamtkohlenhydrate", "ballaststoffe",
        "zucker", "zugesetzter zucker", "vitamin", "mineral", "kalzium", "eisen", "kalium", "vitamin d", "vitamin c",
        # Italian
        "calorie", "proteine", "grassi", "carboidrati", "porzione", "valori nutrizionali", "per porzione",
        "grassi totali", "grassi saturi", "grassi trans", "colesterolo", "sodio", "carboidrati totali", "fibra alimentare",
        "zuccheri", "zuccheri aggiunti", "vitamina", "minerale", "calcio", "ferro", "potassio", "vitamina d", "vitamina c"
    ]
    
    # Apply enhanced smart boosting to ingredient boxes
    for b in ingr_boxes:
        if 'ocr_text' in b and 'ocr_confidence' in b:
            text_lower = b['ocr_text'].lower()
            ocr_conf = b['ocr_confidence']
            
            # Store original confidence before boosting
            original_confidence = b["confidence"]
            
            # Calculate ingredient keyword matches
            ingredient_matches = sum(1 for word in ingredient_keywords if word in text_lower)
            nutrition_matches = sum(1 for word in nutrition_keywords if word in text_lower)
            
            # Base confidence adjustment
            confidence_change = 0.0
            boost_reasons = []
            penalty_reasons = []
            
            # Positive boosts
            if ingredient_matches > 0:
                # More matches = higher boost (up to +0.4)
                ingredient_boost = min(ingredient_matches * 0.1, 0.4)
                confidence_change += ingredient_boost
                boost_reasons.append(f"ingredient keywords (+{ingredient_boost:.1f})")
            
            # High OCR confidence boost
            if ocr_conf > 0.85:
                confidence_change += 0.2
                boost_reasons.append("high OCR confidence (+0.2)")
            elif ocr_conf > 0.7:
                confidence_change += 0.1
                boost_reasons.append("good OCR confidence (+0.1)")
            
            # Text length boost (substantial text = more likely to be ingredients list)
            if len(text_lower) > 100:
                confidence_change += 0.15
                boost_reasons.append("substantial text length (+0.15)")
            elif len(text_lower) > 50:
                confidence_change += 0.1
                boost_reasons.append("moderate text length (+0.1)")
            
            # Negative penalties
            if nutrition_matches > ingredient_matches:
                # More nutrition keywords than ingredient keywords = likely wrong detection
                penalty = min(nutrition_matches * 0.2, 0.6)
                confidence_change -= penalty
                penalty_reasons.append(f"nutrition keywords detected (-{penalty:.1f})")
            
            # Low OCR confidence penalty
            if ocr_conf < 0.5:
                confidence_change -= 0.2
                penalty_reasons.append("low OCR confidence (-0.2)")
            elif ocr_conf < 0.7:
                confidence_change -= 0.1
                penalty_reasons.append("moderate OCR confidence (-0.1)")
            
            # Apply the confidence change
            boosted_confidence = round(max(min(original_confidence + confidence_change, 1.0), 0.0), 3)
            
            # Store both original and boosted confidence
            b["confidence_original"] = original_confidence
            b["confidence_boosted"] = boosted_confidence
            b["confidence"] = boosted_confidence  # Keep the main confidence field for compatibility
            
            # Log the changes
            if boost_reasons or penalty_reasons:
                print(f"🔍 Ingredient box confidence: {original_confidence:.3f} → {boosted_confidence:.3f}")
                if boost_reasons:
                    print(f"  ✅ Boosts: {', '.join(boost_reasons)}")
                if penalty_reasons:
                    print(f"  ❌ Penalties: {', '.join(penalty_reasons)}")
    
    # Apply enhanced smart boosting to nutrition boxes
    for b in nutr_boxes:
        if 'ocr_text' in b and 'ocr_confidence' in b:
            text_lower = b['ocr_text'].lower()
            ocr_conf = b['ocr_confidence']
            
            # Store original confidence before boosting
            original_confidence = b["confidence"]
            
            # Calculate nutrition keyword matches
            nutrition_matches = sum(1 for word in nutrition_keywords if word in text_lower)
            ingredient_matches = sum(1 for word in ingredient_keywords if word in text_lower)
            
            # Base confidence adjustment
            confidence_change = 0.0
            boost_reasons = []
            penalty_reasons = []
            
            # ENHANCED MISCLASSIFICATION DETECTION
            # Check for various forms of "ingredients" in multiple languages
            ingredient_indicators = [
                "ingredients", "ingredientes", "ingrédients", "zutaten", "ingredienti",
                "ngredients", "ngredientes", "ngrédients", "ngredienti"  # OCR artifacts
            ]
            
            has_ingredient_indicator = any(indicator in text_lower for indicator in ingredient_indicators)
            
            # Check for ingredient-like content patterns
            ingredient_patterns = [
                "chilli", "garlic", "sugar", "xanthan", "stabilizer", "stabilisant", "estabilizador",
                "piments", "ail", "sucre", "sel", "gomaxantana", "comoestabilizador"
            ]
            
            has_ingredient_patterns = any(pattern in text_lower for pattern in ingredient_patterns)
            
            # HEAVY PENALTY for nutrition boxes that contain ingredient-like content
            if has_ingredient_indicator or (has_ingredient_patterns and ingredient_matches > nutrition_matches):
                penalty = 0.9  # Very heavy penalty for wrong classification
                confidence_change -= penalty
                penalty_reasons.append(f"ingredients content in nutrition box (-{penalty:.1f})")
                print(f"⚠️  WARNING: Nutrition box contains ingredient-like content - heavily penalized!")
                print(f"   Text: '{text_lower[:100]}...'")
            
            # Positive boosts (only if not heavily penalized for ingredients)
            elif nutrition_matches > 0:
                # More matches = higher boost (up to +0.4)
                nutrition_boost = min(nutrition_matches * 0.1, 0.4)
                confidence_change += nutrition_boost
                boost_reasons.append(f"nutrition keywords (+{nutrition_boost:.1f})")
            
            # High OCR confidence boost
            if ocr_conf > 0.85:
                confidence_change += 0.2
                boost_reasons.append("high OCR confidence (+0.2)")
            elif ocr_conf > 0.7:
                confidence_change += 0.1
                boost_reasons.append("good OCR confidence (+0.1)")
            
            # Numbers and percentages boost (nutrition tables contain lots of numbers)
            number_count = len([c for c in text_lower if c.isdigit()])
            percent_count = text_lower.count('%')
            if number_count > 5 or percent_count > 2:
                confidence_change += 0.15
                boost_reasons.append("nutrition data patterns (+0.15)")
            
            # Negative penalties (only if not already heavily penalized)
            if ingredient_matches > nutrition_matches and not has_ingredient_indicator:
                # More ingredient keywords than nutrition keywords = likely wrong detection
                penalty = min(ingredient_matches * 0.2, 0.6)
                confidence_change -= penalty
                penalty_reasons.append(f"ingredient keywords detected (-{penalty:.1f})")
            
            # Low OCR confidence penalty
            if ocr_conf < 0.5:
                confidence_change -= 0.2
                penalty_reasons.append("low OCR confidence (-0.2)")
            elif ocr_conf < 0.7:
                confidence_change -= 0.1
                penalty_reasons.append("moderate OCR confidence (-0.1)")
            
            # Apply the confidence change
            boosted_confidence = round(max(min(original_confidence + confidence_change, 1.0), 0.0), 3)
            
            # Store both original and boosted confidence
            b["confidence_original"] = original_confidence
            b["confidence_boosted"] = boosted_confidence
            b["confidence"] = boosted_confidence  # Keep the main confidence field for compatibility
            
            # Log the changes
            if boost_reasons or penalty_reasons:
                print(f"🔍 Nutrition box confidence: {original_confidence:.3f} → {boosted_confidence:.3f}")
                if boost_reasons:
                    print(f"  ✅ Boosts: {', '.join(boost_reasons)}")
                if penalty_reasons:
                    print(f"  ❌ Penalties: {', '.join(penalty_reasons)}")
    
    return ingr_boxes, nutr_boxes
