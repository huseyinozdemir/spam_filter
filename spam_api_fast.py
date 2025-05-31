from fastapi import FastAPI, Depends
from pydantic import BaseModel
import joblib
import re
import string
from typing import Protocol, Optional


class TextCleaner(Protocol):
    def clean(self, text: str) -> str: ...


class ModelManager(Protocol):
    def predict(self, text: str) -> tuple[int, str]: ...


class ResponseFormatter(Protocol):
    def format_prediction(self, prediction: int, label: str) -> dict: ...


class SpamTextCleaner:    
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'https?://\S+', '[LINK]', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return text


class SpamModelManager:    
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
    
    def predict(self, text: str) -> tuple[int, str]:
        vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(vectorized)[0]
        label = "Spam" if prediction == 1 else "Ham"
        return int(prediction), label


class StandardResponseFormatter:    
    def format_prediction(self, prediction: int, label: str) -> dict:
        return {
            "prediction": prediction,
            "label": label
        }


class DetailedResponseFormatter:    
    def format_prediction(self, prediction: int, label: str) -> dict:
        return {
            "prediction": prediction,
            "label": label,
            "confidence": float(prediction),
            "is_spam": prediction == 1,
            "status": "success"
        }


# Pydantic models (SRP)
class PredictionRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    prediction: int
    label: str


# DIP - High-level service depends on abstractions
class SpamPredictionService:
    """High-level business logic"""
    
    def __init__(
        self,
        text_cleaner: TextCleaner,
        model_manager: ModelManager,
        response_formatter: ResponseFormatter
    ):
        self.text_cleaner = text_cleaner
        self.model_manager = model_manager
        self.response_formatter = response_formatter
    
    def predict_spam(self, text: str) -> dict:
        """OCP - Open for extension, closed for modification"""
        # Clean text
        cleaned_text = self.text_cleaner.clean(text)
        
        # Make prediction
        prediction, label = self.model_manager.predict(cleaned_text)
        
        # Format response
        return self.response_formatter.format_prediction(prediction, label)


# Dependency injection configuration
class AppConfig:
    """Configuration for dependency injection"""
    
    def __init__(
        self,
        model_path: str = "spam_model.pkl",
        vectorizer_path: str = "vectorizer.pkl",
        use_detailed_response: bool = False
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.use_detailed_response = use_detailed_response


# Factory pattern
class ServiceFactory:    
    @staticmethod
    def create_spam_service(config: AppConfig) -> SpamPredictionService:
        text_cleaner = SpamTextCleaner()
        model_manager = SpamModelManager(config.model_path, config.vectorizer_path)
        
        if config.use_detailed_response:
            response_formatter = DetailedResponseFormatter()
        else:
            response_formatter = StandardResponseFormatter()
        
        return SpamPredictionService(text_cleaner, model_manager, response_formatter)


# Dependency provider
def get_spam_service() -> SpamPredictionService:
    """Dependency injection for FastAPI"""
    config = AppConfig()
    return ServiceFactory.create_spam_service(config)


# FastAPI app with dependency injection
app = FastAPI(title="Spam Detection API", version="1.0.0")


@app.post("/predict")
def predict(
    request: PredictionRequest,
    spam_service: SpamPredictionService = Depends(get_spam_service)
) -> dict:
    """Single responsibility: HTTP endpoint"""
    return spam_service.predict_spam(request.text)


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "1.0.0"}


# Alternative endpoints (OCP - extension)
@app.post("/predict/detailed")
def predict_detailed(
    request: PredictionRequest
) -> dict:
    config = AppConfig(use_detailed_response=True)
    service = ServiceFactory.create_spam_service(config)
    return service.predict_spam(request.text) 