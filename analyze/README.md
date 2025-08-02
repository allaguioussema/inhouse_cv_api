# CV Analysis API with LangChain Integration

## 🚀 New Features (v3.0.0)

### ✨ LangChain Integration
- **Free LLM Providers**: Llama 3, DeepSeek, Mistral, Ollama
- **Smart Model Selection**: Automatic fallback between providers
- **Enhanced Prompt Management**: Structured prompts with LangChain templates
- **Better Error Handling**: Graceful degradation when models are unavailable

### 🌍 Multilingual Support
- **French Language Support**: Complete French analysis capabilities
- **Bilingual Prompts**: Native French prompts for better accuracy
- **Translation Service**: Built-in text translation capabilities
- **Language Detection**: Auto-detect content language

### 🔧 Enhanced Architecture
- **Modular Design**: Clean separation of concerns
- **Type Safety**: Full type hints and Pydantic models
- **Comprehensive Logging**: Detailed processing information
- **Performance Metrics**: Processing time and confidence scores

## 📋 Requirements

```bash
pip install -r requirements.txt
```

### Environment Variables

```bash
# Free LLM API Keys
HUGGINGFACE_API_KEY=your_huggingface_key
FIREWORKS_API_KEY=your_fireworks_key
TOGETHER_API_KEY=your_together_key

# Ollama Configuration
OLLAMA_MODEL=llama2:7b
OLLAMA_URL=http://localhost:11434

# OpenRouter (Optional)
USE_OPENROUTER=false
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=deepseek/deepseek-coder:6.7b
```

## 🏃‍♂️ Quick Start

### 1. Install Dependencies
```bash
cd inhouse_cv_api/analyze
pip install -r requirements.txt
```

### 2. Set up Free LLMs

#### Option A: Local Ollama (Recommended)
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Llama 3
ollama pull llama2:7b

# For better performance, use 13B model
ollama pull llama2:13b
```

#### Option B: HuggingFace (Free)
1. Get API key from https://huggingface.co/settings/tokens
2. Set environment variable: `export HUGGINGFACE_API_KEY=your_key`

#### Option C: Fireworks AI (Free tier)
1. Get API key from https://fireworks.ai/
2. Set environment variable: `export FIREWORKS_API_KEY=your_key`

#### Option D: Together AI (Free tier)
1. Get API key from https://together.ai/
2. Set environment variable: `export TOGETHER_API_KEY=your_key`

### 3. Start the API
```bash
python start_api.py
```

The API will be available at `http://localhost:8001`

### 4. Test the API
```bash
# Health check
curl http://localhost:8001/health

# English analysis
curl -X POST http://localhost:8001/analyze/ingredients \
  -H "Content-Type: application/json" \
  -d '{"text": "Sugar, Salt, Milk, Wheat Flour", "language": "en"}'

# French analysis
curl -X POST http://localhost:8001/analyze/ingredients \
  -H "Content-Type: application/json" \
  -d '{"text": "Sucre, Sel, Lait, Farine de Blé", "language": "fr"}'
```

## 📚 API Endpoints

### Core Analysis Endpoints

#### `POST /analyze/ingredients`
Analyze ingredient lists with multilingual support.

**Request:**
```json
{
  "text": "Sugar, Salt, Milk, Wheat Flour, Artificial Flavors",
  "language": "en"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "allergens": ["milk", "wheat"],
    "additives": ["artificial flavors"],
    "warnings": ["high sugar content"],
    "dietary_restrictions": {
      "vegetarian": true,
      "vegan": false,
      "gluten_free": false,
      "dairy_free": false
    },
    "ingredient_quality": {
      "natural_ingredients": ["salt"],
      "artificial_ingredients": ["artificial flavors"],
      "organic_ingredients": [],
      "processed_ingredients": ["sugar", "wheat flour"]
    },
    "nutritional_highlights": {
      "high_protein": false,
      "high_fiber": false,
      "high_sugar": true,
      "high_salt": true,
      "high_fat": false
    },
    "safety_assessment": {
      "health_score": 4,
      "safety_notes": "Contains common allergens and artificial ingredients",
      "recommendations": "Check for allergies and consider natural alternatives"
    },
    "summary": "Contains allergens and artificial ingredients"
  },
  "language": "en",
  "processing_time": 1.23,
  "confidence": 0.9
}
```

#### `POST /analyze/nutrition`
Analyze nutrition facts with multilingual support.

**Request:**
```json
{
  "text": "Energy: 480kcal, Protein: 8g, Fat: 12g, Carbohydrates: 45g",
  "language": "en"
}
```

#### `POST /analyze/comprehensive`
Comprehensive food analysis combining ingredients and nutrition.

#### `POST /analyze/auto`
Auto-detect content type and analyze accordingly.

### Utility Endpoints

#### `GET /health`
Check service health and available models.

#### `GET /models/available`
Get list of available LLM models.

#### `GET /languages/supported`
Get supported languages.

#### `POST /translate`
Translate text to target language.

#### `GET /example`
Get example requests for testing.

## 🌍 Language Support

### English (en)
- Full ingredient analysis
- Comprehensive nutrition analysis
- Health and safety assessments
- Dietary restriction identification

### French (fr)
- **Analyse complète des ingrédients**
- **Analyse nutritionnelle détaillée**
- **Évaluation de sécurité alimentaire**
- **Identification des restrictions alimentaires**

### Example French Analysis

**Request:**
```json
{
  "text": "Sucre, Sel, Lait, Farine de Blé, Arômes Artificiels",
  "language": "fr"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "allergenes": ["lait", "blé"],
    "additifs": ["arômes artificiels"],
    "avertissements": ["teneur élevée en sucre"],
    "restrictions_alimentaires": {
      "vegetarien": true,
      "vegan": false,
      "sans_gluten": false,
      "sans_lactose": false
    },
    "qualite_ingredients": {
      "ingredients_naturels": ["sel"],
      "ingredients_artificiels": ["arômes artificiels"],
      "ingredients_bio": [],
      "ingredients_transformes": ["sucre", "farine de blé"]
    },
    "points_nutritionnels": {
      "proteines_elevees": false,
      "fibres_elevees": false,
      "sucre_eleve": true,
      "sel_eleve": true,
      "graisse_elevee": false
    },
    "evaluation_securite": {
      "score_sante": 4,
      "notes_securite": "Contient des allergènes courants et des ingrédients artificiels",
      "recommandations": "Vérifiez les allergies et considérez des alternatives naturelles"
    },
    "resume": "Contient des allergènes et des ingrédients artificiels"
  },
  "language": "fr",
  "processing_time": 1.45,
  "confidence": 0.9
}
```

## 🔧 Free LLM Features

### Available Models
1. **Ollama Llama 3** (Local) - Best performance, no API costs
2. **DeepSeek Coder** (HuggingFace) - Excellent for structured analysis
3. **Mistral 7B** (HuggingFace) - Good balance of speed and accuracy
4. **Llama 3 8B** (HuggingFace) - Latest Llama model
5. **Fireworks Llama 3** (Fireworks AI) - Free tier available
6. **Together AI Llama 3** (Together AI) - Free tier available

### Model Management
```python
from .llm_prompter import llm_manager

# Check available models
print(llm_manager.models.keys())

# Use specific model
result = llm_manager.analyze_with_langchain(
    text="Your text here",
    analysis_type="ingredients",
    language=Language.FRENCH
)
```

### Prompt Templates
The system uses LangChain prompt templates optimized for free LLMs:

- `ingredients_en`: English ingredient analysis
- `ingredients_fr`: French ingredient analysis
- `nutrition_en`: English nutrition analysis
- `nutrition_fr`: French nutrition analysis

### Chain Operations
```python
# Create custom chain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze this: {text}"
)

chain = prompt | llm_manager.models["ollama_llama3"] | StrOutputParser()
result = chain.invoke({"text": "Your analysis text"})
```

## 🛠️ Development

### Setting up Free LLMs

#### 1. Local Ollama Setup
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull models
ollama pull llama2:7b
ollama pull llama2:13b  # For better performance
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b
```

#### 2. HuggingFace Setup
```bash
# Get API key
# Visit: https://huggingface.co/settings/tokens

# Set environment variable
export HUGGINGFACE_API_KEY=your_key_here
```

#### 3. Fireworks AI Setup
```bash
# Get API key
# Visit: https://fireworks.ai/

# Set environment variable
export FIREWORKS_API_KEY=your_key_here
```

#### 4. Together AI Setup
```bash
# Get API key
# Visit: https://together.ai/

# Set environment variable
export TOGETHER_API_KEY=your_key_here
```

### Adding New Languages

1. **Add Language Enum:**
```python
class Language(Enum):
    ENGLISH = "en"
    FRENCH = "fr"
    SPANISH = "es"  # New language
```

2. **Create Prompts:**
```python
prompts["ingredients_es"] = PromptTemplate(
    input_variables=["text"],
    template="Tu eres un experto en seguridad alimentaria..."
)
```

3. **Add Fallback Responses:**
```python
if language == Language.SPANISH:
    # Spanish fallback logic
```

### Adding New Models

1. **Initialize in LangChainLLMManager:**
```python
if NEW_API_KEY:
    models["new_model"] = NewLLM(
        api_key=NEW_API_KEY,
        model="new-model"
    )
```

2. **Add to model order:**
```python
model_order = ["ollama_llama3", "deepseek_coder", "new_model", "ollama_legacy"]
```

## 📊 Performance

### Model Performance Comparison
- **Ollama Llama 3 (Local)**: Best performance, no API costs, fastest response
- **DeepSeek Coder**: Excellent for structured analysis, good JSON output
- **Mistral 7B**: Good balance of speed and accuracy
- **Llama 3 8B**: Latest model, excellent reasoning
- **Fireworks/Together AI**: Cloud-based, free tier available
- **Fallback**: Basic keyword-based analysis

### Processing Times
- **Local Ollama**: 1-3 seconds
- **HuggingFace Models**: 2-5 seconds
- **Cloud Models**: 3-8 seconds
- **Fallback Analysis**: <1 second

## 🔍 Error Handling

The system includes comprehensive error handling:

1. **Model Unavailability**: Automatic fallback to next available model
2. **API Rate Limits**: Exponential backoff and retry logic
3. **Invalid Responses**: JSON parsing with fallback responses
4. **Network Issues**: Timeout handling and graceful degradation

## 🧪 Testing

### Run Tests
```bash
pytest tests/
```

### Manual Testing
```bash
# Test English analysis
curl -X POST http://localhost:8001/analyze/ingredients \
  -H "Content-Type: application/json" \
  -d '{"text": "Sugar, Salt, Milk", "language": "en"}'

# Test French analysis
curl -X POST http://localhost:8001/analyze/ingredients \
  -H "Content-Type: application/json" \
  -d '{"text": "Sucre, Sel, Lait", "language": "fr"}'

# Test translation
curl -X POST http://localhost:8001/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "target_language": "fr"}'
```

## 📈 Monitoring

### Health Check
```bash
curl http://localhost:8001/health
```

### Available Models
```bash
curl http://localhost:8001/models/available
```

### Supported Languages
```bash
curl http://localhost:8001/languages/supported
```

## 💰 Cost Comparison

| Provider | Model | Cost | Performance |
|----------|-------|------|-------------|
| Ollama | Llama 3 | Free | ⭐⭐⭐⭐⭐ |
| HuggingFace | DeepSeek/Mistral | Free | ⭐⭐⭐⭐ |
| Fireworks | Llama 3 | Free tier | ⭐⭐⭐⭐ |
| Together AI | Llama 3 | Free tier | ⭐⭐⭐⭐ |
| OpenAI | GPT-4 | $0.03/1K tokens | ⭐⭐⭐⭐⭐ |
| Anthropic | Claude | $0.015/1K tokens | ⭐⭐⭐⭐⭐ |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the health endpoint
2. Review the logs
3. Test with the example endpoint
4. Create an issue with detailed information

---

**🎉 Enjoy the enhanced CV Analysis API with LangChain integration and free LLM support!** 