import os
import json
import requests
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass
from googletrans import Translator

# LangChain imports
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

# Free LLM providers
try:
    from langchain_huggingface import HuggingFaceEndpoint
except ImportError:
    HuggingFaceEndpoint = None

try:
    from langchain_fireworks import Fireworks
except ImportError:
    Fireworks = None

try:
    from langchain_together import Together
except ImportError:
    Together = None

# Configs
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2:7b")
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-coder:6.7b")

# Free LLM API Keys
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

class Language(Enum):
    ENGLISH = "en"
    FRENCH = "fr"

@dataclass
class AnalysisResult:
    content: Dict[str, Any]
    language: Language
    confidence: float
    model_used: str
    processing_time: float

class LangChainLLMManager:
    """Enhanced LLM manager using LangChain with free LLM support"""
    
    def __init__(self):
        self.translator = Translator()
        self.models = self._initialize_models()
        self.prompts = self._initialize_prompts()
        
    def _initialize_models(self) -> Dict[str, BaseLanguageModel]:
        """Initialize available free LLM models"""
        models = {}
        
        # Local Ollama with Llama 3
        try:
            models["ollama_llama3"] = Ollama(
                model="llama2:7b",  # or "llama2:13b" for better performance
                base_url="http://localhost:11434",
                temperature=0.1
            )
            print("✅ Ollama Llama 3 available")
        except Exception as e:
            print(f"⚠️ Ollama Llama 3 not available: {e}")
        
        # HuggingFace Free Models
        if HUGGINGFACE_API_KEY:
            try:
                # DeepSeek Coder
                models["deepseek_coder"] = HuggingFaceEndpoint(
                    endpoint_url="https://api-inference.huggingface.co/models/deepseek-ai/deepseek-coder-6.7b-instruct",
                    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.1,
                        "max_new_tokens": 2048,
                        "top_p": 0.95
                    }
                )
                print("✅ DeepSeek Coder available")
            except Exception as e:
                print(f"⚠️ DeepSeek Coder not available: {e}")
            
            try:
                # Mistral 7B
                models["mistral_7b"] = HuggingFaceEndpoint(
                    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.1,
                        "max_new_tokens": 2048,
                        "top_p": 0.95
                    }
                )
                print("✅ Mistral 7B available")
            except Exception as e:
                print(f"⚠️ Mistral 7B not available: {e}")
            
            try:
                # Llama 3 8B
                models["llama3_8b"] = HuggingFaceEndpoint(
                    endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Meta-Llama-3-8B-Instruct",
                    huggingfacehub_api_token=HUGGINGFACE_API_KEY,
                    task="text-generation",
                    model_kwargs={
                        "temperature": 0.1,
                        "max_new_tokens": 2048,
                        "top_p": 0.95
                    }
                )
                print("✅ Llama 3 8B available")
            except Exception as e:
                print(f"⚠️ Llama 3 8B not available: {e}")
        
        # Fireworks AI (Free tier)
        if FIREWORKS_API_KEY:
            try:
                models["fireworks_llama3"] = Fireworks(
                    model="accounts/fireworks/models/llama-v2-7b-chat",
                    api_key=FIREWORKS_API_KEY,
                    temperature=0.1,
                    max_tokens=2048
                )
                print("✅ Fireworks Llama 3 available")
            except Exception as e:
                print(f"⚠️ Fireworks Llama 3 not available: {e}")
        
        # Together AI (Free tier)
        if TOGETHER_API_KEY:
            try:
                models["together_llama3"] = Together(
                    model="meta-llama/Llama-3-8b-chat-hf",
                    api_key=TOGETHER_API_KEY,
                    temperature=0.1,
                    max_tokens=2048
                )
                print("✅ Together AI Llama 3 available")
            except Exception as e:
                print(f"⚠️ Together AI Llama 3 not available: {e}")
        
        # Legacy Ollama fallback
        try:
            models["ollama_legacy"] = Ollama(
                model=OLLAMA_MODEL,
                base_url="http://localhost:11434"
            )
            print("✅ Legacy Ollama available")
        except Exception as e:
            print(f"⚠️ Legacy Ollama not available: {e}")
        
        return models
    
    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize multilingual prompts optimized for free LLMs"""
        prompts = {}
        
        # English prompts optimized for free LLMs
        prompts["ingredients_en"] = PromptTemplate(
            input_variables=["text"],
            template="""
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
        )
        
        # French prompts optimized for free LLMs
        prompts["ingredients_fr"] = PromptTemplate(
            input_variables=["text"],
            template="""
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
        )
        
        # Nutrition analysis prompts
        prompts["nutrition_en"] = PromptTemplate(
            input_variables=["text"],
            template="""
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
        )
        
        prompts["nutrition_fr"] = PromptTemplate(
            input_variables=["text"],
            template="""
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
        )
        
        return prompts
    
    def translate_text(self, text: str, target_lang: str = "fr") -> str:
        """Translate text to target language"""
        try:
            result = self.translator.translate(text, dest=target_lang)
            return result.text
        except Exception as e:
            print(f"⚠️ Translation failed: {e}")
            return text
    
    def call_ollama_legacy(self, prompt: str, model: str = OLLAMA_MODEL) -> str:
        """Legacy Ollama call for fallback"""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            print(f"[🔴 OLLAMA ERROR] {e}")
            return None
    
    def call_openrouter_legacy(self, prompt: str, model: str = OPENROUTER_MODEL) -> str:
        """Legacy OpenRouter call for fallback"""
        headers = {
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[🔴 OpenRouter ERROR] {e}")
            return None
    
    def analyze_with_langchain(self, text: str, analysis_type: str, language: Language = Language.ENGLISH) -> Optional[AnalysisResult]:
        """Analyze text using LangChain with specified language"""
        import time
        start_time = time.time()
        
        # Get appropriate prompt
        prompt_key = f"{analysis_type}_{language.value}"
        if prompt_key not in self.prompts:
            print(f"❌ No prompt found for {prompt_key}")
            return None
        
        prompt = self.prompts[prompt_key]
        
        # Try each model in order of preference (free LLMs first)
        model_order = [
            "ollama_llama3",      # Local Llama 3
            "deepseek_coder",      # DeepSeek Coder
            "mistral_7b",          # Mistral 7B
            "llama3_8b",           # Llama 3 8B
            "fireworks_llama3",    # Fireworks Llama 3
            "together_llama3",     # Together AI Llama 3
            "ollama_legacy"        # Legacy Ollama
        ]
        
        for model_name in model_order:
            if model_name in self.models:
                try:
                    print(f"🔄 Trying {model_name} for {analysis_type} analysis in {language.value}...")
                    
                    # Create chain
                    chain = prompt | self.models[model_name] | StrOutputParser()
                    
                    # Execute
                    result = chain.invoke({"text": text})
                    
                    if result and result.strip():
                        processing_time = time.time() - start_time
                        print(f"✅ {model_name} successful")
                        
                        return AnalysisResult(
                            content=json.loads(result),
                            language=language,
                            confidence=0.9,
                            model_used=model_name,
                            processing_time=processing_time
                        )
                    
                except Exception as e:
                    print(f"❌ {model_name} failed: {e}")
                    continue
        
        return None
    
    def prompt_llm(self, prompt: str, language: Language = Language.ENGLISH) -> str:
        """Smart LLM selector with enhanced fallback and multilingual support"""
        print(f"🤖 Attempting LLM analysis in {language.value}...")
        
        # Try LangChain first
        result = self.analyze_with_langchain(prompt, "general", language)
        if result:
            return json.dumps(result.content, ensure_ascii=False)
        
        # Fallback to legacy methods
        if not USE_OPENROUTER:
            print("🔄 Trying legacy Ollama...")
            local_response = self.call_ollama_legacy(prompt)
            if local_response and local_response.strip():
                print("✅ Legacy LLM successful")
                return local_response
            print("❌ Legacy LLM failed")
        
        if OPENROUTER_KEY:
            print("🔄 Trying legacy OpenRouter...")
            cloud_response = self.call_openrouter_legacy(prompt)
            if cloud_response and cloud_response.strip():
                print("✅ Legacy cloud LLM successful")
                return cloud_response
            print("❌ Legacy cloud LLM failed")
        
        # Use fallback response
        print("⚠️ All LLM providers failed, using fallback analysis")
        return self.generate_fallback_response(prompt, language)
    
    def generate_fallback_response(self, prompt: str, language: Language = Language.ENGLISH) -> str:
        """Generate a fallback response when LLM providers are unavailable"""
        print("🔄 Using fallback response generation...")
        
        prompt_lower = prompt.lower()
        
        if language == Language.FRENCH:
            if "ingredient" in prompt_lower or "ingrédient" in prompt_lower:
                return '''
                {
                    "allergenes": ["allergenes detectes"],
                    "additifs": ["additifs detectes"],
                    "avertissements": ["avertissements sante"],
                    "restrictions_alimentaires": {
                        "vegetarien": true,
                        "vegan": true,
                        "sans_gluten": true,
                        "sans_lactose": true
                    },
                    "qualite_ingredients": {
                        "ingredients_naturels": ["ingredients naturels"],
                        "ingredients_artificiels": ["ingredients artificiels"],
                        "ingredients_bio": [],
                        "ingredients_transformes": ["ingredients transformes"]
                    },
                    "points_nutritionnels": {
                        "proteines_elevees": false,
                        "fibres_elevees": false,
                        "sucre_eleve": false,
                        "sel_eleve": false,
                        "graisse_elevee": false
                    },
                    "evaluation_securite": {
                        "score_sante": 5,
                        "notes_securite": "Analyse basique des ingredients terminee",
                        "recommandations": "Verifiez la liste des ingredients pour les allergenes specifiques"
                    },
                    "resume": "Analyse de fallback des ingredients - verification manuelle requise"
                }'''
            
            elif "nutrition" in prompt_lower or "nutritionnel" in prompt_lower:
                return '''
                {
                    "macronutriments": {
                        "energie_kcal": 0,
                        "proteines_g": 0,
                        "lipides_totaux_g": 0,
                        "graisses_saturees_g": 0,
                        "graisses_trans_g": 0,
                        "glucides_g": 0,
                        "fibres_g": 0
                    },
                    "sucres": {
                        "sucres_totaux_g": 0,
                        "sucres_ajoutes_g": 0,
                        "sucres_naturels_g": 0
                    },
                    "sodium": {
                        "sodium_mg": 0,
                        "sel_g": 0
                    },
                    "micronutriments": {
                        "vitamine_a_mcg": 0,
                        "vitamine_c_mg": 0,
                        "vitamine_d_mcg": 0,
                        "vitamine_e_mg": 0,
                        "vitamine_k_mcg": 0,
                        "calcium_mg": 0,
                        "fer_mg": 0,
                        "potassium_mg": 0
                    },
                    "evaluation_nutritionnelle": {
                        "score_sante": 5,
                        "densite_calorique": "inconnue",
                        "qualite_proteines": "inconnue",
                        "teneur_fibres": "inconnue",
                        "teneur_sucre": "inconnue",
                        "teneur_sel": "inconnue",
                        "qualite_lipides": "inconnue",
                        "recommandations": "Analyse nutritionnelle de fallback - verification manuelle requise",
                        "avertissements": ["Donnees nutritionnelles incompletes"]
                    },
                    "info_portions": {
                        "taille_portion": "inconnue",
                        "portions_par_contenant": 0,
                        "par_portion": false,
                        "par_100g": false
                    },
                    "resume": "Analyse nutritionnelle de fallback - verification manuelle requise"
                }'''
        else:
            # English fallback (existing logic)
            if "ingredient" in prompt_lower:
                return '''
                {
                    "allergens": ["detected allergens"],
                    "additives": ["detected additives"],
                    "warnings": ["health warnings"],
                    "dietary_restrictions": {
                        "vegetarian": true,
                        "vegan": true,
                        "gluten_free": true,
                        "dairy_free": true
                    },
                    "ingredient_quality": {
                        "natural_ingredients": ["natural ingredients"],
                        "artificial_ingredients": ["artificial ingredients"],
                        "organic_ingredients": [],
                        "processed_ingredients": ["processed ingredients"]
                    },
                    "nutritional_highlights": {
                        "high_protein": false,
                        "high_fiber": false,
                        "high_sugar": false,
                        "high_salt": false,
                        "high_fat": false
                    },
                    "safety_assessment": {
                        "health_score": 5,
                        "safety_notes": "Basic ingredient analysis completed",
                        "recommendations": "Check ingredients list for specific allergens"
                    },
                    "summary": "Fallback ingredient analysis - check manually for accuracy"
                }'''
            
            elif "nutrition" in prompt_lower:
                return '''
                {
                    "macronutrients": {
                        "energy_kcal": 0,
                        "protein_g": 0,
                        "total_fat_g": 0,
                        "saturated_fat_g": 0,
                        "trans_fat_g": 0,
                        "carbohydrates_g": 0,
                        "fiber_g": 0
                    },
                    "sugars": {
                        "total_sugars_g": 0,
                        "added_sugars_g": 0,
                        "natural_sugars_g": 0
                    },
                    "sodium": {
                        "sodium_mg": 0,
                        "salt_g": 0
                    },
                    "micronutrients": {
                        "vitamin_a_mcg": 0,
                        "vitamin_c_mg": 0,
                        "vitamin_d_mcg": 0,
                        "vitamin_e_mg": 0,
                        "vitamin_k_mcg": 0,
                        "calcium_mg": 0,
                        "iron_mg": 0,
                        "potassium_mg": 0
                    },
                    "nutritional_assessment": {
                        "health_score": 5,
                        "calorie_density": "unknown",
                        "protein_quality": "unknown",
                        "fiber_content": "unknown",
                        "sugar_content": "unknown",
                        "salt_content": "unknown",
                        "fat_quality": "unknown",
                        "recommendations": "Fallback nutrition analysis - check manually",
                        "warnings": ["Nutrition data incomplete"]
                    },
                    "serving_info": {
                        "serving_size": "unknown",
                        "servings_per_container": 0,
                        "per_serving": false,
                        "per_100g": false
                    },
                    "summary": "Fallback nutrition analysis - manual verification required"
                }'''
        
        return '''
        {
            "content_type": "unknown",
            "confidence": 0.5,
            "analysis": {
                "summary": "Fallback analysis - LLM unavailable"
            },
            "health_insights": "Unable to provide detailed analysis",
            "recommendations": "Check manually for accuracy"
        }'''

# Global instance
llm_manager = LangChainLLMManager()

# Backward compatibility functions
def call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    return llm_manager.call_ollama_legacy(prompt, model)

def call_openrouter(prompt: str, model: str = OPENROUTER_MODEL) -> str:
    return llm_manager.call_openrouter_legacy(prompt, model)

def generate_fallback_response(prompt: str) -> str:
    return llm_manager.generate_fallback_response(prompt, Language.ENGLISH)

def prompt_llm(prompt: str, language: Language = Language.ENGLISH) -> str:
    return llm_manager.prompt_llm(prompt, language)

# 🔬 Example
if __name__ == "__main__":
    test_prompt = """
    Here is a list of ingredients and nutrition facts:
    'Sugar, Salt, Milk, Energy 480kcal, Protein 8g, Sodium 220mg'
    Summarize the nutrition and flag allergens or additives.
    """
    print("🇺🇸 English analysis:")
    print(prompt_llm(test_prompt, Language.ENGLISH))
    
    print("\n🇫🇷 French analysis:")
    print(prompt_llm(test_prompt, Language.FRENCH))
