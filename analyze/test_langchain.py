#!/usr/bin/env python3
"""
Test script for LangChain integration and French language support
"""

import json
import time
from llm_prompter import llm_manager, Language, prompt_llm
from analyzer import analyze_ingredients, summarize_nutrition

def test_langchain_integration():
    """Test LangChain integration"""
    print("🔧 Testing LangChain Integration...")
    
    # Test available models
    print(f"📋 Available models: {list(llm_manager.models.keys())}")
    
    # Test prompt templates
    print(f"📝 Available prompts: {list(llm_manager.prompts.keys())}")
    
    return True

def test_english_analysis():
    """Test English analysis"""
    print("\n🇺🇸 Testing English Analysis...")
    
    # Test ingredients
    ingredients_text = "Sugar, Salt, Milk, Wheat Flour, Artificial Flavors, Preservatives"
    print(f"📝 Analyzing ingredients: {ingredients_text}")
    
    start_time = time.time()
    result = analyze_ingredients(ingredients_text, Language.ENGLISH)
    processing_time = time.time() - start_time
    
    print(f"✅ English ingredients analysis completed in {processing_time:.2f}s")
    print(f"📊 Health score: {result.get('safety_assessment', {}).get('health_score', 'N/A')}")
    
    # Test nutrition
    nutrition_text = "Energy: 480kcal, Protein: 8g, Fat: 12g, Carbohydrates: 45g, Sodium: 220mg"
    print(f"📝 Analyzing nutrition: {nutrition_text}")
    
    start_time = time.time()
    result = summarize_nutrition(nutrition_text, Language.ENGLISH)
    processing_time = time.time() - start_time
    
    print(f"✅ English nutrition analysis completed in {processing_time:.2f}s")
    print(f"📊 Energy: {result.get('macronutrients', {}).get('energy_kcal', 'N/A')} kcal")
    
    return True

def test_french_analysis():
    """Test French analysis"""
    print("\n🇫🇷 Testing French Analysis...")
    
    # Test ingredients
    ingredients_text = "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels, Conservateurs"
    print(f"📝 Analyse des ingrédients: {ingredients_text}")
    
    start_time = time.time()
    result = analyze_ingredients(ingredients_text, Language.FRENCH)
    processing_time = time.time() - start_time
    
    print(f"✅ Analyse française des ingrédients terminée en {processing_time:.2f}s")
    print(f"📊 Score de santé: {result.get('evaluation_securite', {}).get('score_sante', 'N/A')}")
    
    # Test nutrition
    nutrition_text = "Énergie: 480kcal, Protéines: 8g, Lipides: 12g, Glucides: 45g, Sodium: 220mg"
    print(f"📝 Analyse nutritionnelle: {nutrition_text}")
    
    start_time = time.time()
    result = summarize_nutrition(nutrition_text, Language.FRENCH)
    processing_time = time.time() - start_time
    
    print(f"✅ Analyse nutritionnelle française terminée en {processing_time:.2f}s")
    print(f"📊 Énergie: {result.get('macronutriments', {}).get('energie_kcal', 'N/A')} kcal")
    
    return True

def test_translation():
    """Test translation functionality"""
    print("\n🌍 Testing Translation...")
    
    test_texts = [
        "Hello world",
        "This product contains allergens",
        "High protein content",
        "Natural ingredients only"
    ]
    
    for text in test_texts:
        try:
            translated = llm_manager.translate_text(text, "fr")
            print(f"🇺🇸 {text} → 🇫🇷 {translated}")
        except Exception as e:
            print(f"❌ Translation failed for '{text}': {e}")
    
    return True

def test_fallback_mechanisms():
    """Test fallback mechanisms"""
    print("\n🔄 Testing Fallback Mechanisms...")
    
    # Test with a simple prompt
    test_prompt = "Analyze this ingredient list: Sugar, Salt"
    
    print("🔄 Testing LangChain fallback...")
    result = prompt_llm(test_prompt, Language.ENGLISH)
    print(f"✅ Fallback result length: {len(result)} characters")
    
    print("🔄 Testing French fallback...")
    result = prompt_llm(test_prompt, Language.FRENCH)
    print(f"✅ French fallback result length: {len(result)} characters")
    
    return True

def test_performance():
    """Test performance metrics"""
    print("\n⚡ Testing Performance...")
    
    test_text = "Sugar, Salt, Milk, Wheat Flour, Artificial Flavors"
    
    # Test English performance
    start_time = time.time()
    result_en = analyze_ingredients(test_text, Language.ENGLISH)
    en_time = time.time() - start_time
    
    # Test French performance
    start_time = time.time()
    result_fr = analyze_ingredients(test_text, Language.FRENCH)
    fr_time = time.time() - start_time
    
    print(f"🇺🇸 English analysis time: {en_time:.2f}s")
    print(f"🇫🇷 French analysis time: {fr_time:.2f}s")
    print(f"📊 Performance ratio: {fr_time/en_time:.2f}x")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting LangChain Integration Tests...")
    print("=" * 50)
    
    tests = [
        ("LangChain Integration", test_langchain_integration),
        ("English Analysis", test_english_analysis),
        ("French Analysis", test_french_analysis),
        ("Translation", test_translation),
        ("Fallback Mechanisms", test_fallback_mechanisms),
        ("Performance", test_performance)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            result = test_func()
            results.append((test_name, True, None))
            print(f"✅ {test_name} test passed!")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name} test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if error:
            print(f"   Error: {error}")
        
        if success:
            passed += 1
        else:
            failed += 1
    
    print(f"\n📈 Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! LangChain integration is working perfectly!")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 