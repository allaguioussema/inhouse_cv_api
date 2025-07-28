# InHouse CV API

A comprehensive computer vision microservices architecture for food packaging analysis, featuring object detection, OCR, classification, and AI-powered reasoning.

## 🏗️ Architecture Overview

```
inhouse_cv_api/
│
├── gateway/                  # Central API gateway (FastAPI + NGINX proxy)
├── detect/                   # Object detection (YOLOv11L)
│   ├── models/               # nutrition_table.pt, ingredient_block.pt
│   └── ...
├── ocr/                      # OCR via PaddleOCR
├── classify/                 # Region classification (ResNet / CLIP)
├── analyze/                  # LLM-based reasoning (GPT / Claude / Mistral)
├── similarity/               # Image similarity engine (CLIP + Faiss)
├── feedback/                 # Box correction loop + LabelStudio integration
│
├── shared/                   # Common utilities: image tools, schema, logger
│
├── docker-compose.yml        # Deploy all services together
├── .env                      # Central config file
└── README.md                 # Full project documentation
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.9+
- CUDA-compatible GPU (optional, for faster inference)

### Installation

1. **Clone and setup:**
```bash
git clone <repository-url>
cd inhouse_cv_api
```

2. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. **Deploy with Docker:**
```bash
docker-compose up -d
```

4. **Access the API:**
- Gateway: http://localhost:8000
- Swagger UI: http://localhost:8000/docs

## 📋 Services

### 🔍 Detection Service (`detect/`)
- **Purpose**: Object detection for nutrition tables and ingredient blocks
- **Models**: YOLOv11L with custom training
- **Endpoints**:
  - `POST /detect/nutrition` - Detect nutrition information tables
  - `POST /detect/ingredients` - Detect ingredient lists
  - `POST /detect/all` - Detect all relevant regions

### 📝 OCR Service (`ocr/`)
- **Purpose**: Text extraction from detected regions
- **Engine**: PaddleOCR with custom dictionary
- **Endpoints**:
  - `POST /ocr/extract` - Extract text from image regions
  - `POST /ocr/table` - Extract structured table data

### 🏷️ Classification Service (`classify/`)
- **Purpose**: Region classification and content type identification
- **Models**: ResNet + CLIP for zero-shot classification
- **Endpoints**:
  - `POST /classify/region` - Classify detected regions
  - `POST /classify/content` - Identify content types

### 🧠 Analysis Service (`analyze/`)
- **Purpose**: AI-powered reasoning and data interpretation
- **Models**: GPT-4, Claude, Mistral
- **Endpoints**:
  - `POST /analyze/nutrition` - Analyze nutrition facts
  - `POST /analyze/ingredients` - Analyze ingredient safety
  - `POST /analyze/summary` - Generate comprehensive report

### 🔍 Similarity Service (`similarity/`)
- **Purpose**: Image similarity search and duplicate detection
- **Engine**: CLIP embeddings + Faiss index
- **Endpoints**:
  - `POST /similarity/search` - Find similar images
  - `POST /similarity/duplicates` - Detect duplicate products

### 📊 Feedback Service (`feedback/`)
- **Purpose**: User feedback collection and model retraining
- **Integration**: LabelStudio for annotation
- **Endpoints**:
  - `POST /feedback/correction` - Submit box corrections
  - `GET /feedback/status` - Check retraining status

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=cv_api
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Paths
NUTRITION_MODEL_PATH=./detect/models/nutrition_table.pt
INGREDIENT_MODEL_PATH=./detect/models/ingredient_block.pt

# Service Ports
GATEWAY_PORT=8000
DETECT_PORT=8001
OCR_PORT=8002
CLASSIFY_PORT=8003
ANALYZE_PORT=8004
SIMILARITY_PORT=8005
FEEDBACK_PORT=8006
```

## 📊 API Usage Examples

### Detect Nutrition Information
```python
import requests

# Upload image and detect nutrition table
with open('food_package.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/detect/nutrition', files=files)
    
detections = response.json()
print(f"Found {len(detections['boxes'])} nutrition regions")
```

### Extract and Analyze Text
```python
# Extract text from detected regions
ocr_response = requests.post('http://localhost:8000/ocr/extract', 
                           json={'image_path': 'food_package.jpg', 
                                'regions': detections['boxes']})

# Analyze nutrition facts
analysis = requests.post('http://localhost:8000/analyze/nutrition',
                        json={'text': ocr_response.json()['text']})
```

## 🐳 Docker Deployment

### Single Service
```bash
cd detect
docker build -t cv-detect .
docker run -p 8001:8001 cv-detect
```

### Full Stack
```bash
docker-compose up -d
```

### Production Deployment
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 🔄 Development Workflow

### Adding New Models
1. Place model files in appropriate `models/` directory
2. Update service configuration
3. Add new endpoints in service `main.py`
4. Update gateway routes
5. Test with sample images

### Model Retraining Pipeline
1. Collect corrections via feedback service
2. Export dataset from LabelStudio
3. Run training script with new data
4. Validate model performance
5. Deploy updated model

## 📈 Monitoring & Logging

- **Metrics**: Prometheus + Grafana
- **Logging**: Structured JSON logs
- **Health Checks**: `/health` endpoints on all services
- **Performance**: Response time tracking

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: GitHub Issues
- **Documentation**: `/docs` endpoints
- **API Reference**: Swagger UI at `/docs`

## 🔮 Roadmap

- [ ] Multi-language OCR support
- [ ] Real-time video processing
- [ ] Edge deployment optimization
- [ ] Advanced analytics dashboard
- [ ] Mobile SDK integration 