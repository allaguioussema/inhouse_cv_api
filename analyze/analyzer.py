from .llm_prompter import prompt_llm, llm_manager, Language
from typing import Dict, Any, Optional
import json

def analyze_ingredients(text: str, language: Language = Language.ENGLISH) -> dict:
    """
    Analyze ingredients with multilingual support
    """
    if language == Language.FRENCH:
        prompt = f"""
        Vous êtes un expert en sécurité alimentaire et nutritionniste. Analysez cette liste d'ingrédients avec une haute précision:

        TEXTE: {text}

        Effectuez une analyse complète:

        1. ALLERGÈNES: Identifiez tous les allergènes (lait, œufs, poisson, crustacés, noix, arachides, blé, soja, sésame)
        2. ADDITIFS: Listez tous les additifs, conservateurs, colorants, arômes, stabilisants, émulsifiants
        3. AVERTISSEMENTS SANTÉ: Signalez l'excès de sucre, sel, graisse, ingrédients artificiels, ou additifs préoccupants
        4. QUALITÉ DES INGRÉDIENTS: Évaluez si les ingrédients sont naturels, bio, transformés, ou artificiels
        5. RESTRICTIONS ALIMENTAIRES: Identifiez si adapté pour végétarien, végan, sans gluten, sans lactose
        6. APERÇUS NUTRITIONNELS: Mettez en évidence les protéines/fibres/vitamines/minéraux élevés/faibles
        7. ÉVALUATION DE SÉCURITÉ: Notez la santé globale (1-10) et fournissez des notes de sécurité

        Retournez un JSON détaillé:
        {{
            "allergenes": ["liste", "des", "allergenes"],
            "additifs": ["liste", "des", "additifs"],
            "avertissements": ["avertissements", "sante"],
            "restrictions_alimentaires": {{
                "vegetarien": true/false,
                "vegan": true/false,
                "sans_gluten": true/false,
                "sans_lactose": true/false
            }},
            "qualite_ingredients": {{
                "ingredients_naturels": ["liste"],
                "ingredients_artificiels": ["liste"],
                "ingredients_bio": ["liste"],
                "ingredients_transformes": ["liste"]
            }},
            "points_nutritionnels": {{
                "proteines_elevees": true/false,
                "fibres_elevees": true/false,
                "sucre_eleve": true/false,
                "sel_eleve": true/false,
                "graisse_elevee": true/false
            }},
            "evaluation_securite": {{
                "score_sante": 1-10,
                "notes_securite": "evaluation detaillee de la securite",
                "recommandations": "recommandations de sante"
            }},
            "resume": "resume complet de l'analyse"
        }}

        Soyez extrêmement minutieux et précis. Considérez les réglementations de sécurité alimentaire et les directives de santé.
        """
    else:
        prompt = f"""
        You are a food safety expert and nutritionist. Analyze this ingredient list with high precision:

        TEXT: {text}

        Perform a comprehensive analysis:

        1. ALLERGENS: Identify all allergens (milk, eggs, fish, shellfish, tree nuts, peanuts, wheat, soybeans, sesame)
        2. ADDITIVES: List all additives, preservatives, colorings, flavorings, stabilizers, emulsifiers
        3. HEALTH WARNINGS: Flag excessive sugar, salt, fat, artificial ingredients, or concerning additives
        4. INGREDIENT QUALITY: Assess if ingredients are natural, organic, processed, or artificial
        5. DIETARY RESTRICTIONS: Identify if suitable for vegetarian, vegan, gluten-free, dairy-free diets
        6. NUTRITIONAL INSIGHTS: Highlight high/low protein, fiber, vitamins, minerals
        7. SAFETY ASSESSMENT: Rate overall healthiness (1-10) and provide safety notes

        Return detailed JSON:
        {{
            "allergens": ["list", "of", "allergens"],
            "additives": ["list", "of", "additives"],
            "warnings": ["health", "warnings"],
            "dietary_restrictions": {{
                "vegetarian": true/false,
                "vegan": true/false,
                "gluten_free": true/false,
                "dairy_free": true/false
            }},
            "ingredient_quality": {{
                "natural_ingredients": ["list"],
                "artificial_ingredients": ["list"],
                "organic_ingredients": ["list"],
                "processed_ingredients": ["list"]
            }},
            "nutritional_highlights": {{
                "high_protein": true/false,
                "high_fiber": true/false,
                "high_sugar": true/false,
                "high_salt": true/false,
                "high_fat": true/false
            }},
            "safety_assessment": {{
                "health_score": 1-10,
                "safety_notes": "detailed safety assessment",
                "recommendations": "health recommendations"
            }},
            "summary": "comprehensive analysis summary"
        }}

        Be extremely thorough and accurate. Consider food safety regulations and health guidelines.
        """
    
    response = prompt_llm(prompt, language)
    return parse_llm_json(response)

def summarize_nutrition(text: str, language: Language = Language.ENGLISH) -> dict:
    """
    Analyze nutrition with multilingual support
    """
    if language == Language.FRENCH:
        prompt = f"""
        Vous êtes un diététicien agréé et expert en nutrition. Analysez ces informations nutritionnelles avec précision:

        TEXTE: {text}

        Extrayez et analysez les données nutritionnelles de manière complète:

        1. MACRONUTRIMENTS: Énergie (kcal), Protéines (g), Lipides (g), Glucides (g), Fibres (g)
        2. MICRONUTRIMENTS: Vitamines (A, C, D, E, K, B-complexe), Minéraux (Fer, Calcium, Sodium, Potassium)
        3. SUCRES: Sucres totaux, sucres ajoutés, sucres naturels
        4. LIPIDES: Lipides totaux, graisses saturées, graisses trans, graisses insaturées
        5. SODIUM: Teneur en sel et niveaux de sodium
        6. ANALYSE DES PORTIONS: Par portion vs par 100g/100ml
        7. ÉVALUATION NUTRITIONNELLE: Note de santé, recommandations, avertissements

        Retournez un JSON complet:
        {{
            "macronutriments": {{
                "energie_kcal": number,
                "proteines_g": number,
                "lipides_totaux_g": number,
                "graisses_saturees_g": number,
                "graisses_trans_g": number,
                "glucides_g": number,
                "fibres_g": number
            }},
            "sucres": {{
                "sucres_totaux_g": number,
                "sucres_ajoutes_g": number,
                "sucres_naturels_g": number
            }},
            "sodium": {{
                "sodium_mg": number,
                "sel_g": number
            }},
            "micronutriments": {{
                "vitamine_a_mcg": number,
                "vitamine_c_mg": number,
                "vitamine_d_mcg": number,
                "vitamine_e_mg": number,
                "vitamine_k_mcg": number,
                "calcium_mg": number,
                "fer_mg": number,
                "potassium_mg": number
            }},
            "evaluation_nutritionnelle": {{
                "score_sante": 1-10,
                "densite_calorique": "faible/moyenne/elevee",
                "qualite_proteines": "excellente/bonne/moyenne/faible",
                "teneur_fibres": "excellente/bonne/moyenne/faible",
                "teneur_sucre": "faible/moyenne/elevee",
                "teneur_sel": "faible/moyenne/elevee",
                "qualite_lipides": "excellente/bonne/moyenne/faible",
                "recommandations": "recommandations detaillees",
                "avertissements": ["avertissements nutritionnels"]
            }},
            "info_portions": {{
                "taille_portion": "string",
                "portions_par_contenant": number,
                "par_portion": boolean,
                "par_100g": boolean
            }},
            "resume": "resume nutritionnel complet"
        }}
        """
    else:
        prompt = f"""
        You are a registered dietitian and nutrition expert. Analyze this nutrition information with precision:

        TEXT: {text}

        Extract and analyze nutritional data comprehensively:

        1. MACRONUTRIENTS: Energy (kcal), Protein (g), Fat (g), Carbohydrates (g), Fiber (g)
        2. MICRONUTRIENTS: Vitamins (A, C, D, E, K, B-complex), Minerals (Iron, Calcium, Sodium, Potassium)
        3. SUGARS: Total sugars, added sugars, natural sugars
        4. FATS: Total fat, saturated fat, trans fat, unsaturated fats
        5. SODIUM: Salt content and sodium levels
        6. SERVING ANALYSIS: Per serving vs per 100g/100ml calculations
        7. NUTRITIONAL ASSESSMENT: Health rating, recommendations, warnings

        Return comprehensive JSON:
        {{
            "macronutrients": {{
                "energy_kcal": number,
                "protein_g": number,
                "total_fat_g": number,
                "saturated_fat_g": number,
                "trans_fat_g": number,
                "carbohydrates_g": number,
                "fiber_g": number
            }},
            "sugars": {{
                "total_sugars_g": number,
                "added_sugars_g": number,
                "natural_sugars_g": number
            }},
            "sodium": {{
                "sodium_mg": number,
                "salt_g": number
            }},
            "micronutrients": {{
                "vitamin_a_mcg": number,
                "vitamin_c_mg": number,
                "vitamin_d_mcg": number,
                "vitamin_e_mg": number,
                "vitamin_k_mcg": number,
                "calcium_mg": number,
                "iron_mg": number,
                "potassium_mg": number
            }},
            "nutritional_assessment": {{
                "health_score": 1-10,
                "calorie_density": "low/medium/high",
                "protein_quality": "excellent/good/fair/poor",
                "fiber_content": "excellent/good/fair/poor",
                "sugar_content": "low/medium/high",
                "salt_content": "low/medium/high",
                "fat_quality": "excellent/good/fair/poor",
                "recommendations": "detailed recommendations",
                "warnings": ["nutrition warnings"]
            }},
            "serving_info": {{
                "serving_size": "string",
                "servings_per_container": number,
                "per_serving": boolean,
                "per_100g": boolean
            }},
            "summary": "comprehensive nutrition summary"
        }}
        """
    
    response = prompt_llm(prompt, language)
    return parse_llm_json(response)

def analyze_food_package_comprehensive(text: str, detection_type: str) -> dict:
    """
    Smart analysis that adapts based on content type and provides comprehensive insights
    """
    if detection_type == "ingredient":
        return analyze_ingredients(text)
    elif detection_type == "nutrition":
        return summarize_nutrition(text)
    else:
        # Smart detection - analyze both ingredient and nutrition aspects
        prompt = f"""
        You are a food safety expert and nutritionist. Analyze this food packaging text comprehensively:

        TEXT: {text}

        Determine if this is primarily an ingredients list, nutrition facts, or mixed content.
        Then provide appropriate analysis:

        If it's INGREDIENTS-focused:
        - Allergen detection
        - Additive analysis
        - Health warnings
        - Dietary restrictions

        If it's NUTRITION-focused:
        - Macronutrient extraction
        - Micronutrient analysis
        - Health assessment
        - Serving information

        If it's MIXED content:
        - Combine both analyses
        - Cross-reference information
        - Provide comprehensive insights

        Return JSON with:
        {{
            "content_type": "ingredients/nutrition/mixed",
            "confidence": 0.0-1.0,
            "analysis": {{
                // Appropriate analysis based on content type
            }},
            "cross_references": {{
                // If mixed content, cross-reference ingredients with nutrition
            }},
            "health_insights": "comprehensive health analysis",
            "recommendations": "detailed recommendations"
        }}

        Be extremely thorough and accurate.
        """
        response = prompt_llm(prompt)
        return parse_llm_json(response)

def detect_food_content_type(text: str) -> dict:
    """
    Smart detection of whether text is ingredients, nutrition, or mixed content
    """
    prompt = f"""
    Analyze this food packaging text and determine its primary content type:

    TEXT: {text}

    Classify as:
    1. "ingredients" - if primarily lists ingredients, allergens, additives
    2. "nutrition" - if primarily shows nutrition facts, calories, macronutrients
    3. "mixed" - if contains both ingredients and nutrition information
    4. "other" - if doesn't fit the above categories

    Return JSON:
    {{
        "content_type": "ingredients/nutrition/mixed/other",
        "confidence": 0.0-1.0,
        "reasoning": "explanation of classification",
        "key_indicators": {{
            "ingredient_indicators": ["list of ingredient-related words"],
            "nutrition_indicators": ["list of nutrition-related words"],
            "mixed_indicators": ["list of mixed content indicators"]
        }}
    }}
    """
    response = prompt_llm(prompt)
    return parse_llm_json(response)

def detect_food_content_type_fallback(text: str) -> dict:
    """
    Fallback content type detection using keyword analysis when LLM is unavailable
    """
    text_lower = text.lower()
    
    # Keyword indicators
    ingredient_keywords = [
        "ingredients", "ingredientes", "ingrédients", "zutaten", "ingredienti",
        "contains", "may contain", "allergen", "additive", "preservative",
        "chilli", "garlic", "sugar", "xanthan", "stabilizer", "emulsifier"
    ]
    
    nutrition_keywords = [
        "nutrition", "calories", "energy", "protein", "fat", "carbohydrate",
        "sugar", "fiber", "sodium", "salt", "vitamin", "mineral",
        "per serving", "per 100g", "per 100ml", "nutrition facts"
    ]
    
    # Count matches
    ingredient_matches = sum(1 for word in ingredient_keywords if word in text_lower)
    nutrition_matches = sum(1 for word in nutrition_keywords if word in text_lower)
    
    # Determine content type
    if ingredient_matches > nutrition_matches and ingredient_matches > 0:
        content_type = "ingredients"
        confidence = min(0.8, 0.5 + (ingredient_matches * 0.1))
        reasoning = f"Contains {ingredient_matches} ingredient-related keywords"
    elif nutrition_matches > ingredient_matches and nutrition_matches > 0:
        content_type = "nutrition"
        confidence = min(0.8, 0.5 + (nutrition_matches * 0.1))
        reasoning = f"Contains {nutrition_matches} nutrition-related keywords"
    elif ingredient_matches > 0 and nutrition_matches > 0:
        content_type = "mixed"
        confidence = 0.7
        reasoning = "Contains both ingredient and nutrition keywords"
    else:
        content_type = "other"
        confidence = 0.5
        reasoning = "No clear ingredient or nutrition indicators found"
    
    return {
        "content_type": content_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "key_indicators": {
            "ingredient_indicators": [word for word in ingredient_keywords if word in text_lower],
            "nutrition_indicators": [word for word in nutrition_keywords if word in text_lower],
            "mixed_indicators": []
        },
        "fallback_analysis": True
    }

def parse_llm_json(raw: str) -> dict:
    try:
        import json
        import re
        
        # Clean the response - remove comments and fix common JSON issues
        cleaned = raw.strip()
        
        # Remove single-line comments (// ...)
        cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)
        
        # Remove multi-line comments (/* ... */)
        cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
        
        # Fix common JSON issues
        cleaned = re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
        cleaned = re.sub(r',\s*]', ']', cleaned)  # Remove trailing commas in arrays
        
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            # Try to parse the JSON
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If JSON is malformed, try to fix common issues
                print(f"⚠️ JSON parsing failed: {e}")
                print(f"Raw JSON: {json_str[:200]}...")
                
                # Try to extract key-value pairs manually
                manual_parse = {}
                # Look for patterns like "key": value
                kv_pattern = r'"([^"]+)"\s*:\s*([^,\n}]+)'
                matches = re.findall(kv_pattern, json_str)
                
                for key, value in matches:
                    # Clean the value
                    value = value.strip().strip('"').strip("'")
                    # Try to convert to number if possible
                    try:
                        if '.' in value:
                            manual_parse[key] = float(value)
                        else:
                            manual_parse[key] = int(value)
                    except ValueError:
                        manual_parse[key] = value
                
                if manual_parse:
                    return {
                        "parsed_manually": True,
                        "data": manual_parse,
                        "original_error": str(e)
                    }
                else:
                    return {
                        "error": f"Failed to parse JSON: {str(e)}",
                        "raw": raw,
                        "parsed": False
                    }
        else:
            # If no JSON found, return the raw response with some structure
            return {
                "raw_response": raw,
                "parsed": False,
                "summary": "LLM response could not be parsed as JSON"
            }
    except Exception as e:
        return {
            "error": f"Failed to parse LLM response: {str(e)}", 
            "raw": raw,
            "parsed": False
        }
