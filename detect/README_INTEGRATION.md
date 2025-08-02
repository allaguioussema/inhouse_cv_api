# 🔗 Detect + Analyze Integration Guide

## 🎯 **What is Connected?**

The **detect** and **analyze** modules are now fully integrated! This means:

- 📸 **Detect**: Takes images and finds food packaging areas (ingredients + nutrition)
- 🔤 **OCR**: Extracts text from those areas using PaddleOCR
- 🤖 **Analyze**: Uses free AI models to analyze the extracted text
- 🌍 **Multilingual**: Supports English and French analysis

## 🚀 **Quick Start**

### **Step 1: Install Dependencies**
```bash
cd detect
pip install -r requirements.txt
```

### **Step 2: Start the API**
```bash
python main.py
```

### **Step 3: Test the Integration**
```bash
python test_integration.py
```

## 📋 **Available Endpoints**

### **Main Detection Endpoints**

| Endpoint | What it does | Parameters |
|----------|-------------|------------|
| `POST /detect/custom` | Full analysis with AI | `file`, `detection_mode`, `preferred_language`, `use_llm` |
| `POST /detect/single` | Single image analysis | `file`, `use_llm` |
| `POST /detect/full-batch` | Multiple images | `files`, `use_llm` |
| `POST /camera/capture-and-detect` | Camera + analysis | `detection_mode`, `preferred_language`, `use_llm` |

### **Info Endpoints**

| Endpoint | What it returns |
|----------|----------------|
| `GET /` | API status and features |
| `GET /health` | Health check with analyze status |
| `GET /analyze/status` | Analyze module capabilities |
| `GET /llm/analysis/info` | LLM analysis capabilities |

## 🎯 **How to Use**

### **Example 1: Basic Food Analysis**
```bash
curl -X POST http://localhost:8000/detect/custom \
  -F "file=@food_image.jpg" \
  -F "detection_mode=both" \
  -F "preferred_language=en" \
  -F "use_llm=true"
```

### **Example 2: French Analysis**
```bash
curl -X POST http://localhost:8000/detect/custom \
  -F "file=@food_image.jpg" \
  -F "detection_mode=ingredient" \
  -F "preferred_language=fr" \
  -F "use_llm=true"
```

### **Example 3: Camera Capture + Analysis**
```bash
curl -X POST http://localhost:8000/camera/capture-and-detect \
  -F "detection_mode=both" \
  -F "preferred_language=en" \
  -F "use_llm=true"
```

## 📊 **What You Get Back**

### **Sample Response Structure**
```json
{
  "success": true,
  "detection_results": {
    "ingredient_boxes": [
      {
        "bbox": [x, y, width, height],
        "confidence": 0.95,
        "text": "Ingredients: Sugar, Salt, Milk...",
        "llm_analysis": {
          "analysis_type": "enhanced_ingredient_analysis",
          "allergens": ["milk", "wheat", "soy"],
          "additives": ["artificial flavors", "preservatives"],
          "dietary_restrictions": {
            "vegetarian": false,
            "vegan": false,
            "gluten_free": false,
            "dairy_free": false
          },
          "safety_assessment": {
            "health_score": 4,
            "safety_notes": "Contains common allergens",
            "recommendations": "Check for allergies"
          }
        }
      }
    ],
    "nutrition_boxes": [
      {
        "bbox": [x, y, width, height],
        "confidence": 0.92,
        "text": "Nutrition Facts: Calories: 480...",
        "llm_analysis": {
          "analysis_type": "enhanced_nutrition_analysis",
          "macronutrients": {
            "energy_kcal": 480,
            "protein_g": 8,
            "total_fat_g": 12,
            "carbohydrates_g": 65
          },
          "nutritional_assessment": {
            "health_score": 6,
            "calorie_density": "high",
            "sugar_content": "high",
            "recommendations": "Moderate consumption"
          }
        }
      }
    ]
  },
  "processing_time": 3.45,
  "analyze_module_available": true,
  "preferred_language": "en"
}
```

## 🌍 **Multilingual Support**

### **English Analysis**
- Detects allergens: milk, wheat, soy, etc.
- Identifies additives: preservatives, colorings, etc.
- Provides health scores (1-10)
- Gives dietary restrictions

### **French Analysis**
- Détecte les allergènes: lait, blé, soja, etc.
- Identifie les additifs: conservateurs, colorants, etc.
- Fournit des scores de santé (1-10)
- Donne les restrictions alimentaires

## 🤖 **Free AI Models Available**

### **Local Models (Ollama)**
- **Llama 3**: `llama2:7b`, `llama2:13b`
- **Mistral**: `mistral:7b`
- **DeepSeek**: `deepseek-coder:6.7b`

### **Cloud Models (Free Tier)**
- **HuggingFace**: DeepSeek Coder, Mistral 7B, Llama 3 8B
- **Fireworks AI**: Llama 3 models
- **Together AI**: Llama 3 models

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Ollama (local)
OLLAMA_MODEL=llama2:7b

# HuggingFace (free)
HUGGINGFACE_API_KEY=your_key_here

# Fireworks AI (free tier)
FIREWORKS_API_KEY=your_key_here

# Together AI (free tier)
TOGETHER_API_KEY=your_key_here
```

### **API Parameters**
- `detection_mode`: `"both"`, `"ingredient"`, `"nutrition"`
- `preferred_language`: `"en"`, `"fr"`
- `use_llm`: `true`/`false` (enable/disable AI analysis)

## 🛠️ **Setup Free AI Models**

### **Option 1: Local Ollama (Recommended)**
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Download models
ollama pull llama2:7b
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b
```

### **Option 2: Cloud Models**
```bash
# Get free API keys
# HuggingFace: https://huggingface.co/settings/tokens
# Fireworks: https://fireworks.ai/
# Together: https://together.ai/

# Set environment variables
export HUGGINGFACE_API_KEY=your_key
export FIREWORKS_API_KEY=your_key
export TOGETHER_API_KEY=your_key
```

## 🧪 **Testing**

### **Run Integration Test**
```bash
python test_integration.py
```

### **Test API Endpoints**
```bash
# Check health
curl http://localhost:8000/health

# Check analyze status
curl http://localhost:8000/analyze/status

# Test with sample image
curl -X POST http://localhost:8000/detect/custom \
  -F "file=@test_image.jpg" \
  -F "use_llm=true"
```

## 🎯 **Use Cases**

### **1. Food Safety Check**
- Upload food packaging image
- Get allergen warnings
- Check for harmful additives
- Receive health recommendations

### **2. Nutritional Analysis**
- Analyze nutrition facts
- Get macronutrient breakdown
- Receive health score
- Get consumption recommendations

### **3. Dietary Compliance**
- Check vegetarian/vegan status
- Identify gluten-free options
- Find dairy-free alternatives
- Get dietary restrictions

### **4. Multilingual Support**
- Analyze French food packaging
- Get localized health insights
- Understand foreign ingredients
- Receive language-specific recommendations

## 🔧 **Troubleshooting**

### **Common Issues**

1. **Analyze module not available**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama not working**
   ```bash
   # Check if Ollama is running
   ollama list
   
   # Start Ollama service
   ollama serve
   ```

3. **API keys not working**
   ```bash
   # Check environment variables
   echo $HUGGINGFACE_API_KEY
   echo $FIREWORKS_API_KEY
   echo $TOGETHER_API_KEY
   ```

4. **Camera not working**
   ```bash
   # Try different camera endpoints
   # Check camera permissions
   # Use file upload instead
   ```

## 🚀 **Next Steps**

1. **Start the API**: `python main.py`
2. **Test with images**: Upload food packaging photos
3. **Check results**: Get comprehensive analysis
4. **Customize**: Modify prompts or add new languages
5. **Scale**: Deploy to production

---

**🎉 You now have a complete food analysis system with computer vision + AI!** 