#!/usr/bin/env python3
"""
Demo script showcasing French language capabilities
"""

import json
from llm_prompter import llm_manager, Language
from analyzer import analyze_ingredients, summarize_nutrition

def demo_french_ingredients():
    """Demo French ingredient analysis"""
    print("🇫🇷 DÉMO: Analyse des Ingrédients en Français")
    print("=" * 50)
    
    # French ingredient examples
    french_ingredients = [
        "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels, Conservateurs",
        "Huile d'Olive Extra Vierge, Tomates Bio, Basilic Frais, Ail, Sel de Mer",
        "Chocolat Noir 70%, Beurre de Cacao, Sucre de Canne, Vanille Bourbon",
        "Saumon Sauvage, Citron, Aneth, Poivre Noir, Sel Rose de l'Himalaya"
    ]
    
    for i, ingredients in enumerate(french_ingredients, 1):
        print(f"\n📝 Exemple {i}: {ingredients}")
        print("-" * 40)
        
        try:
            result = analyze_ingredients(ingredients, Language.FRENCH)
            
            # Display key results
            allergenes = result.get('allergenes', [])
            additifs = result.get('additifs', [])
            score = result.get('evaluation_securite', {}).get('score_sante', 'N/A')
            resume = result.get('resume', 'N/A')
            
            print(f"🔍 Allergènes détectés: {', '.join(allergenes) if allergenes else 'Aucun'}")
            print(f"🧪 Additifs identifiés: {', '.join(additifs) if additifs else 'Aucun'}")
            print(f"📊 Score de santé: {score}/10")
            print(f"📝 Résumé: {resume}")
            
        except Exception as e:
            print(f"❌ Erreur d'analyse: {e}")

def demo_french_nutrition():
    """Demo French nutrition analysis"""
    print("\n🇫🇷 DÉMO: Analyse Nutritionnelle en Français")
    print("=" * 50)
    
    # French nutrition examples
    french_nutrition = [
        "Énergie: 480kcal, Protéines: 8g, Lipides: 12g, Glucides: 45g, Sodium: 220mg",
        "Calories: 120kcal, Protéines: 15g, Graisses: 2g, Glucides: 8g, Fibres: 3g",
        "Énergie: 320kcal, Protéines: 22g, Lipides: 18g, Glucides: 12g, Calcium: 300mg",
        "Calories: 85kcal, Protéines: 1g, Graisses: 0g, Glucides: 20g, Vitamine C: 45mg"
    ]
    
    for i, nutrition in enumerate(french_nutrition, 1):
        print(f"\n📊 Exemple {i}: {nutrition}")
        print("-" * 40)
        
        try:
            result = summarize_nutrition(nutrition, Language.FRENCH)
            
            # Display key results
            energie = result.get('macronutriments', {}).get('energie_kcal', 'N/A')
            proteines = result.get('macronutriments', {}).get('proteines_g', 'N/A')
            lipides = result.get('macronutriments', {}).get('lipides_totaux_g', 'N/A')
            glucides = result.get('macronutriments', {}).get('glucides_g', 'N/A')
            score = result.get('evaluation_nutritionnelle', {}).get('score_sante', 'N/A')
            resume = result.get('resume', 'N/A')
            
            print(f"⚡ Énergie: {energie} kcal")
            print(f"💪 Protéines: {proteines}g")
            print(f"🫒 Lipides: {lipides}g")
            print(f"🍞 Glucides: {glucides}g")
            print(f"📊 Score nutritionnel: {score}/10")
            print(f"📝 Résumé: {resume}")
            
        except Exception as e:
            print(f"❌ Erreur d'analyse: {e}")

def demo_translation():
    """Demo translation capabilities"""
    print("\n🌍 DÉMO: Capacités de Traduction")
    print("=" * 50)
    
    # English texts to translate
    english_texts = [
        "This product contains common allergens",
        "High protein content for muscle building",
        "Natural ingredients only, no artificial preservatives",
        "Low calorie option for weight management",
        "Rich in vitamins and minerals"
    ]
    
    for text in english_texts:
        try:
            translated = llm_manager.translate_text(text, "fr")
            print(f"🇺🇸 {text}")
            print(f"🇫🇷 {translated}")
            print("-" * 30)
        except Exception as e:
            print(f"❌ Traduction échouée pour '{text}': {e}")

def demo_comparison():
    """Demo English vs French comparison"""
    print("\n🔄 DÉMO: Comparaison Anglais vs Français")
    print("=" * 50)
    
    # Same text in both languages
    english_text = "Sugar, Salt, Milk, Wheat Flour, Artificial Flavors"
    french_text = "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels"
    
    print("🇺🇸 Analyse en Anglais:")
    print(f"Texte: {english_text}")
    try:
        result_en = analyze_ingredients(english_text, Language.ENGLISH)
        score_en = result_en.get('safety_assessment', {}).get('health_score', 'N/A')
        print(f"Score de santé: {score_en}/10")
    except Exception as e:
        print(f"❌ Erreur: {e}")
    
    print("\n🇫🇷 Analyse en Français:")
    print(f"Texte: {french_text}")
    try:
        result_fr = analyze_ingredients(french_text, Language.FRENCH)
        score_fr = result_fr.get('evaluation_securite', {}).get('score_sante', 'N/A')
        print(f"Score de santé: {score_fr}/10")
    except Exception as e:
        print(f"❌ Erreur: {e}")

def demo_api_endpoints():
    """Demo API endpoint examples"""
    print("\n🔌 DÉMO: Exemples d'Endpoints API")
    print("=" * 50)
    
    print("📋 Endpoints disponibles:")
    print("• POST /analyze/ingredients - Analyse des ingrédients")
    print("• POST /analyze/nutrition - Analyse nutritionnelle")
    print("• POST /analyze/comprehensive - Analyse complète")
    print("• POST /analyze/auto - Détection automatique")
    print("• GET /health - État du service")
    print("• GET /models/available - Modèles disponibles")
    print("• GET /languages/supported - Langues supportées")
    print("• POST /translate - Traduction de texte")
    
    print("\n📝 Exemple de requête (ingrédients français):")
    print("""
curl -X POST http://localhost:8001/analyze/ingredients \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels",
    "language": "fr"
  }'
    """)
    
    print("📝 Exemple de requête (nutrition française):")
    print("""
curl -X POST http://localhost:8001/analyze/nutrition \\
  -H "Content-Type: application/json" \\
  -d '{
    "text": "Énergie: 480kcal, Protéines: 8g, Lipides: 12g, Glucides: 45g",
    "language": "fr"
  }'
    """)

def main():
    """Run all demos"""
    print("🚀 DÉMONSTRATION: Capacités Françaises du CV Analysis API")
    print("=" * 60)
    print("✨ Intégration LangChain + Support Multilingue")
    print("=" * 60)
    
    demos = [
        ("Analyse des Ingrédients", demo_french_ingredients),
        ("Analyse Nutritionnelle", demo_french_nutrition),
        ("Traduction", demo_translation),
        ("Comparaison EN/FR", demo_comparison),
        ("Endpoints API", demo_api_endpoints)
    ]
    
    for demo_name, demo_func in demos:
        try:
            demo_func()
            print(f"\n✅ {demo_name} - Terminé avec succès!")
        except Exception as e:
            print(f"\n❌ {demo_name} - Erreur: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 DÉMONSTRATION TERMINÉE!")
    print("=" * 60)
    print("📚 Pour plus d'informations, consultez le README.md")
    print("🧪 Pour tester: python test_langchain.py")
    print("🚀 Pour démarrer l'API: python main.py")

if __name__ == "__main__":
    main() 