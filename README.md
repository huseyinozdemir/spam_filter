# 🛡️ Enterprise Spam Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![SOLID](https://img.shields.io/badge/Architecture-SOLID-green.svg)](https://en.wikipedia.org/wiki/SOLID)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![Flask](https://img.shields.io/badge/API-Flask-blue.svg)](https://flask.palletsprojects.com)

**Production-ready spam detection system with enterprise-grade architecture, designed for Postfix integration and modern web APIs.**

## 🎯 Overview

This project implements a **machine learning-based spam detection system** following **SOLID principles** for maximum maintainability, testability, and extensibility. The system supports multiple deployment scenarios including direct Postfix integration and RESTful API services.

## ✨ Features

### 🏗️ **SOLID Architecture**
- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Extensible without modifying existing code
- **Liskov Substitution**: Interface implementations are interchangeable
- **Interface Segregation**: Small, focused protocols
- **Dependency Inversion**: High-level modules depend on abstractions

### 🚀 **Multiple Deployment Options**
- **Postfix Integration**: Direct email processing pipeline
- **FastAPI**: Modern async API with automatic documentation
- **Flask**: Traditional web API for broader compatibility
- **Standalone Training**: Configurable ML pipeline

### 🧠 **Machine Learning**
- **TF-IDF Vectorization**: Advanced text feature extraction
- **Logistic Regression**: Balanced classification for imbalanced datasets
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Reproducible Training**: Configurable random states

### 🔧 **Developer Experience**
- **Protocol-based Design**: Type-safe interfaces
- **Factory Patterns**: Easy dependency injection
- **Configuration Management**: Environment-specific settings
- **Comprehensive Testing**: Mock-friendly architecture

## 📁 Project Structure

```
spam_detection/
├── 🧠 ML Core
│   ├── spam_training.py          # SOLID training pipeline
│   ├── spam_model.pkl           # Trained model
│   └── vectorizer.pkl           # Text vectorizer
├── 🌐 API Services
│   ├── spam_api_fast.py         # FastAPI implementation
│   ├── spam_api_flask.py        # Flask implementation
│   └── spam_api_solid.py        # SOLID FastAPI version
├── 📧 Email Processing
│   ├── check_email_without_service.py  # Direct processing
│   └── check_email.py                  # API-based processing
├── 🧪 Testing
│   ├── test_email_ham.txt       # Normal email samples
│   ├── test_email_spam.txt      # Spam email samples
│   └── test_email_checker.py    # Automated tests
└── 📊 Data
    └── spam_dataset.csv         # Training dataset
```

## 🚀 Quick Start

### 1. **Installation**
```bash
# Clone the repository
git clone <repository-url>
cd spam_detection

# Install dependencies
pip install -r requirements.txt
```

### 2. **Train the Model**
```bash
# Train with default configuration
python spam_training.py

# Train with mock data (for testing)
# Uncomment: config = TrainingConfig(use_mock_data=True)

# Train with advanced text cleaning
# Uncomment: config = TrainingConfig(use_advanced_cleaning=True)
```

### 3. **Start API Services**

#### FastAPI (Recommended)
```bash
# Start FastAPI server
uvicorn spam_api_solid:app --reload --port 8000

# Access interactive docs
open http://localhost:8000/docs
```

#### Flask
```bash
# Start Flask server
python spam_api_flask.py

# API available at http://localhost:5050
```

### 4. **Test the APIs**

#### cURL Examples
```bash
# Test normal message
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Meeting tomorrow at 2 PM"}'

# Test spam message
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "CONGRATULATIONS! Win $1000000 now!"}'

# Detailed response
curl -X POST "http://localhost:8000/predict/detailed" \
     -H "Content-Type: application/json" \
     -d '{"text": "Free money click here!"}'
```

#### Python Client
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Your message here"}
)
print(response.json())
# Output: {"prediction": 0, "label": "Ham"}
```

## 📧 Postfix Integration

### Direct Processing
```bash
# Process email from stdin
cat test_email_spam.txt | python check_email_without_service.py
```

### API-based Processing
```bash
# Process via API (requires running API server)
cat test_email_spam.txt | python check_email.py
```

### Postfix Configuration
Add to `/etc/postfix/main.cf`:
```bash
content_filter = spam-filter:dummy
```

Add to `/etc/postfix/master.cf`:
```bash
spam-filter unix -       n       n       -       -       pipe
  user=filter argv=/path/to/check_email_without_service.py
```

## 🧪 Testing

### Automated Testing
```bash
# Run comprehensive tests
python test_email_checker.py
```

### Manual Testing
```bash
# Test normal email
cat test_email_ham.txt | python check_email_without_service.py

# Test spam email
cat test_email_spam.txt | python check_email_without_service.py
```

## ⚙️ Configuration

### Training Configuration
```python
config = TrainingConfig(
    data_file="spam_dataset.csv",
    test_size=0.2,
    random_state=42,
    ngram_range=(1, 2),
    use_mock_data=False,
    use_advanced_cleaning=False
)
```

### API Configuration
```python
config = AppConfig(
    model_path="spam_model.pkl",
    vectorizer_path="vectorizer.pkl",
    use_detailed_response=False
)
```

## 🏗️ Architecture Highlights

### SOLID Principles Implementation

#### Single Responsibility
```python
class SpamTextCleaner:      # Only text cleaning
class SpamModelManager:     # Only model operations
class EmailSpamProcessor:   # Only email processing coordination
```

#### Dependency Inversion
```python
class EmailSpamProcessor:
    def __init__(self, text_cleaner: TextCleaner, model_manager: ModelManager):
        # Depends on abstractions, not concretions
```

#### Open/Closed Principle
```python
# Add new text cleaner without modifying existing code
class AdvancedTextCleaner:
    def clean(self, text: str) -> str:
        # Extended functionality
```

## 📊 Performance Metrics

The model achieves the following performance on balanced datasets:
- **Accuracy**: ~95%
- **Precision**: ~94%
- **Recall**: ~96%
- **F1-Score**: ~95%

## 🔧 Development

### Code Quality
- **Black**: Code formatting (125 char line length)
- **Flake8**: Linting and style enforcement
- **Type Hints**: Full typing support
- **Protocol-based**: Interface segregation

### VS Code Settings
The project includes `.vscode/settings.json` with:
- Auto-formatting on save
- Import organization
- Linting configuration

## 🤝 Contributing

1. Follow SOLID principles
2. Add type hints for all functions
3. Write tests for new features
4. Use Protocol-based interfaces
5. Maintain configuration flexibility

## 📄 License

This project is licensed under the MIT License.

## 🔗 API Documentation

### FastAPI
- **Interactive Docs**: `http://localhost:8000/docs`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

### Endpoints
- `POST /predict` - Basic spam prediction
- `POST /predict/detailed` - Detailed spam prediction with confidence
- `GET /health` - Health check endpoint

### Response Formats

#### Standard Response
```json
{
  "prediction": 1,
  "label": "Spam"
}
```

#### Detailed Response
```json
{
  "prediction": 1,
  "label": "Spam",
  "confidence": 1.0,
  "is_spam": true,
  "status": "success"
}
```

---

**Built with ❤️ using SOLID principles and modern Python practices**
