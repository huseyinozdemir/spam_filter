from flask import Flask, request, jsonify
import joblib
import re
import string
from typing import Protocol, Optional, Dict, Any


class TextCleaner(Protocol):
    def clean(self, text: str) -> str: ...


class ModelManager(Protocol):
    def predict(self, text: str) -> tuple[int, str]: ...


class RequestValidator(Protocol):
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]: ...


class ResponseFormatter(Protocol):
    def success_response(self, prediction: int, label: str) -> Dict[str, Any]: ...
    def error_response(self, message: str, status_code: int = 400) -> tuple[Dict[str, Any], int]: ...


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


class SpamRequestValidator:
    def validate(self, data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        if not data:
            return False, "No data provided"

        if "text" not in data:
            return False, "No text field provided"

        if not isinstance(data["text"], str):
            return False, "Text field must be a string"

        if not data["text"].strip():
            return False, "Text field cannot be empty"

        return True, None


class StandardResponseFormatter:
    def success_response(self, prediction: int, label: str) -> Dict[str, Any]:
        return {
            "prediction": prediction,
            "label": label
        }

    def error_response(self, message: str, status_code: int = 400) -> tuple[Dict[str, Any], int]:
        return {"error": message}, status_code


class DetailedResponseFormatter:
    def success_response(self, prediction: int, label: str) -> Dict[str, Any]:
        return {
            "prediction": prediction,
            "label": label,
            "confidence": float(prediction),
            "is_spam": prediction == 1,
            "status": "success"
        }

    def error_response(self, message: str, status_code: int = 400) -> tuple[Dict[str, Any], int]:
        return {
            "error": message,
            "status": "error",
            "code": status_code
        }, status_code


class SpamPredictionService:
    """High-level business logic"""

    def __init__(
        self,
        text_cleaner: TextCleaner,
        model_manager: ModelManager,
        validator: RequestValidator,
        formatter: ResponseFormatter
    ):
        self.text_cleaner = text_cleaner
        self.model_manager = model_manager
        self.validator = validator
        self.formatter = formatter

    def predict_spam(self, request_data: Dict[str, Any]) -> tuple[Dict[str, Any], int]:
        # Validate request
        is_valid, error_message = self.validator.validate(request_data)
        if not is_valid:
            return self.formatter.error_response(error_message)

        # Clean text
        cleaned_text = self.text_cleaner.clean(request_data["text"])

        # Make prediction
        prediction, label = self.model_manager.predict(cleaned_text)

        # Format success response
        response = self.formatter.success_response(prediction, label)
        return response, 200


# Configuration for dependency injection
class AppConfig:
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
        validator = SpamRequestValidator()

        if config.use_detailed_response:
            formatter = DetailedResponseFormatter()
        else:
            formatter = StandardResponseFormatter()

        return SpamPredictionService(text_cleaner, model_manager, validator, formatter)


# Flask application factory
def create_app(config: Optional[AppConfig] = None) -> Flask:
    if config is None:
        config = AppConfig()

    app = Flask(__name__)

    spam_service = ServiceFactory.create_spam_service(config)

    @app.route("/predict", methods=["POST"])
    def predict():
        """Single responsibility: HTTP endpoint handling"""
        try:
            data = request.json
            response, status_code = spam_service.predict_spam(data)
            return jsonify(response), status_code
        except Exception as e:
            return jsonify({"error": f"Internal server error {e}"}), 500

    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "version": "1.0.0"})

    # OCP - Extension: Detailed endpoint
    @app.route("/predict/detailed", methods=["POST"])
    def predict_detailed():
        """Detailed prediction endpoint"""
        try:
            detailed_config = AppConfig(
                model_path=config.model_path,
                vectorizer_path=config.vectorizer_path,
                use_detailed_response=True
            )
            detailed_service = ServiceFactory.create_spam_service(detailed_config)

            data = request.json
            response, status_code = detailed_service.predict_spam(data)
            return jsonify(response), status_code
        except Exception as e:
            return jsonify({"error": f"Internal server error {e}"}), 500

    return app


# Create app with default config
app = create_app()


if __name__ == "__main__":
    app.run(port=5050, debug=True)
