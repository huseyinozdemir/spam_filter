#!/usr/bin/env python3
"""SOLID principles uyumlu spam model training pipeline"""

import re
import string
import joblib
import pandas as pd
from typing import Protocol, Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class DataLoader(Protocol):
    def load_data(self) -> pd.DataFrame: ...


class TextCleaner(Protocol):
    def clean(self, text: str) -> str: ...


class FeatureExtractor(Protocol):
    def fit_transform(self, texts: list) -> Any: ...
    def transform(self, texts: list) -> Any: ...


class ModelTrainer(Protocol):
    def train(self, X, y) -> Any: ...


class ModelEvaluator(Protocol):
    def evaluate(self, model: Any, X_test, y_test) -> Dict[str, float]: ...


class ModelSaver(Protocol):
    def save_model(self, model: Any, vectorizer: Any) -> None: ...


class CSVDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)


class MockDataLoader:
    def load_data(self) -> pd.DataFrame:
        data = {
            "text": [
                "Congratulations! You have won a prize. Click here: [LINK]",
                "Click here to claim your prize: [LINK]",
                "You win: [LINK]",
                "Meeting tomorrow at 10 o'clock.",
                "See you tomorrow.",
                "Are you attending the meeting today?",
            ],
            "label": [1, 1, 1, 0, 0, 0]
        }
        return pd.DataFrame(data)


class SpamTextCleaner:
    def clean(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Replace URLs with placeholder
        text = re.sub(r'https?://\S+', '[LINK]', text)
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        return text


class AdvancedTextCleaner:
    def clean(self, text: str) -> str:
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'https?://\S+', '[LINK]', text)
        # Additional: Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        return text


class TfidfFeatureExtractor:
    def __init__(self, ngram_range: Tuple[int, int] = (1, 2)):
        # min_df: Let's eliminate very rare words (2 instead of 1, adjust according to your sample number)
        # max_df: Let's eliminate very common words
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=2, max_df=0.8)

    def fit_transform(self, texts: list) -> Any:
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts: list) -> Any:
        return self.vectorizer.transform(texts)

    def get_vectorizer(self):
        return self.vectorizer


class LogisticRegressionTrainer:
    def __init__(self, class_weight: str = 'balanced', random_state: int = 42):
        self.class_weight = class_weight
        self.random_state = random_state

    def train(self, X, y) -> LogisticRegression:
        model = LogisticRegression(
            class_weight=self.class_weight,
            random_state=self.random_state
        )
        model.fit(X, y)
        return model


class SpamModelEvaluator:
    def evaluate(self, model: Any, X_test, y_test) -> Dict[str, float]:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        return {
            "accuracy": accuracy,
            "precision": report['weighted avg']['precision'],
            "recall": report['weighted avg']['recall'],
            "f1_score": report['weighted avg']['f1-score']
        }


class JoblibModelSaver:
    def __init__(self, model_path: str = "spam_model.pkl", vectorizer_path: str = "vectorizer.pkl"):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

    def save_model(self, model: Any, vectorizer: Any) -> None:
        joblib.dump(model, self.model_path)
        joblib.dump(vectorizer, self.vectorizer_path)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")


# Configuration for training pipeline
class TrainingConfig:
    def __init__(
        self,
        data_file: str = "spam_dataset.csv",
        test_size: float = 0.2,
        random_state: int = 42,
        ngram_range: Tuple[int, int] = (1, 2),
        model_path: str = "spam_model.pkl",
        vectorizer_path: str = "vectorizer.pkl",
        use_mock_data: bool = False,
        use_advanced_cleaning: bool = False
    ):
        self.data_file = data_file
        self.test_size = test_size
        self.random_state = random_state
        self.ngram_range = ngram_range
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.use_mock_data = use_mock_data
        self.use_advanced_cleaning = use_advanced_cleaning


class SpamTrainingPipeline:
    """High-level training pipeline orchestrator"""

    def __init__(
        self,
        data_loader: DataLoader,
        text_cleaner: TextCleaner,
        feature_extractor: FeatureExtractor,
        model_trainer: ModelTrainer,
        model_evaluator: ModelEvaluator,
        model_saver: ModelSaver,
        config: TrainingConfig
    ):
        self.data_loader = data_loader
        self.text_cleaner = text_cleaner
        self.feature_extractor = feature_extractor
        self.model_trainer = model_trainer
        self.model_evaluator = model_evaluator
        self.model_saver = model_saver
        self.config = config

    def run_training(self) -> Dict[str, float]:
        print("🚀 Starting spam detection model training...")

        # Load data
        print("📂 Loading data...")
        df = self.data_loader.load_data()
        print(f"📊 Loaded {len(df)} samples")

        # Clean text
        print("🧹 Cleaning text...")
        df['clean_text'] = df['text'].apply(self.text_cleaner.clean)

        # Extract features
        print("🔧 Extracting features...")
        X = self.feature_extractor.fit_transform(df['clean_text'].tolist())
        y = df['label']

        # Split data
        print("✂️ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )

        # Train model
        print("🎓 Training model...")
        model = self.model_trainer.train(X_train, y_train)

        # Evaluate model
        print("📈 Evaluating model...")
        metrics = self.model_evaluator.evaluate(model, X_test, y_test)

        # Save model
        print("💾 Saving model...")
        vectorizer = self.feature_extractor.get_vectorizer()
        self.model_saver.save_model(model, vectorizer)

        # Test prediction
        self._test_prediction(model, vectorizer)

        print("✅ Training completed!")
        return metrics

    def _test_prediction(self, model, vectorizer):
        test_text = "Congratulations! You have won a prize. Click here"
        cleaned_test = self.text_cleaner.clean(test_text)
        vectorized_test = vectorizer.transform([cleaned_test])
        prediction = model.predict(vectorized_test)

        print("\n🧪 Test prediction:")
        print(f"Text: {test_text}")
        print(f"Prediction: {'Spam' if prediction[0] == 1 else 'Ham'}")


class TrainingPipelineFactory:
    @staticmethod
    def create_training_pipeline(config: TrainingConfig) -> SpamTrainingPipeline:
        # Data loader
        if config.use_mock_data:
            data_loader = MockDataLoader()
        else:
            data_loader = CSVDataLoader(config.data_file)

        # Text cleaner
        if config.use_advanced_cleaning:
            text_cleaner = AdvancedTextCleaner()
        else:
            text_cleaner = SpamTextCleaner()

        # Feature extractor
        feature_extractor = TfidfFeatureExtractor(config.ngram_range)

        # Model trainer
        model_trainer = LogisticRegressionTrainer()

        # Model evaluator
        model_evaluator = SpamModelEvaluator()

        # Model saver
        model_saver = JoblibModelSaver(config.model_path, config.vectorizer_path)

        return SpamTrainingPipeline(
            data_loader=data_loader,
            text_cleaner=text_cleaner,
            feature_extractor=feature_extractor,
            model_trainer=model_trainer,
            model_evaluator=model_evaluator,
            model_saver=model_saver,
            config=config
        )


def main():
    # Default configuration
    config = TrainingConfig()

    # Alternative configurations:
    # config = TrainingConfig(use_mock_data=True)  # For testing without CSV
    # config = TrainingConfig(use_advanced_cleaning=True)  # Advanced text cleaning

    # Create and run training pipeline
    pipeline = TrainingPipelineFactory.create_training_pipeline(config)
    metrics = pipeline.run_training()

    # Print results
    print("\n📊 Training Results:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")


if __name__ == "__main__":
    main()
