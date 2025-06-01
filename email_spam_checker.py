#!/usr/bin/env python3
import sys
import email
from email.policy import default
import joblib
import re
import string
from typing import Protocol


class TextCleaner(Protocol):
    def clean(self, text: str) -> str: ...


class ModelManager(Protocol):
    def predict(self, text: str) -> int: ...


class EmailReader(Protocol):
    def read(self) -> email.message.EmailMessage: ...


class EmailWriter(Protocol):
    def write(self, msg: email.message.EmailMessage) -> None: ...


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

    def predict(self, text: str) -> int:
        vectorized = self.vectorizer.transform([text])
        return self.model.predict(vectorized)[0]


class StdinEmailReader:
    def read(self) -> email.message.EmailMessage:
        raw_email = sys.stdin.read()
        return email.message_from_string(raw_email, policy=default)


class StdoutEmailWriter:
    def write(self, msg: email.message.EmailMessage) -> None:
        sys.stdout.write(msg.as_string())


class EmailBodyExtractor:
    def extract(self, msg: email.message.EmailMessage) -> str:
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body += part.get_content()
        else:
            body = msg.get_content()
        return body


class SpamSubjectModifier:
    def modify_if_spam(self, msg: email.message.EmailMessage, is_spam: bool) -> None:
        if is_spam:
            subject = msg["Subject"] or ""
            if "---SPAM---" not in subject:
                msg.replace_header("Subject", f"---SPAM--- {subject}")


# DIP - High level module depends on abstractions
class EmailSpamProcessor:
    def __init__(
        self,
        email_reader: EmailReader,
        email_writer: EmailWriter,
        text_cleaner: TextCleaner,
        model_manager: ModelManager
    ):
        self.email_reader = email_reader
        self.email_writer = email_writer
        self.text_cleaner = text_cleaner
        self.model_manager = model_manager
        self.body_extractor = EmailBodyExtractor()
        self.subject_modifier = SpamSubjectModifier()

    def process(self) -> None:
        """OCP - Open for extension, closed for modification"""
        # Read email
        msg = self.email_reader.read()

        # Extract body
        body = self.body_extractor.extract(msg)

        # Clean text
        cleaned_text = self.text_cleaner.clean(body)

        # Predict
        prediction = self.model_manager.predict(cleaned_text)
        is_spam = (prediction == 1)

        # Modify subject if spam
        self.subject_modifier.modify_if_spam(msg, is_spam)

        # Write email
        self.email_writer.write(msg)


# Factory for dependency injection
class EmailProcessorFactory:
    @staticmethod
    def create_default_processor() -> EmailSpamProcessor:
        return EmailSpamProcessor(
            email_reader=StdinEmailReader(),
            email_writer=StdoutEmailWriter(),
            text_cleaner=SpamTextCleaner(),
            model_manager=SpamModelManager("spam_model.pkl", "vectorizer.pkl")
        )


def main():
    """Main function using dependency injection"""
    processor = EmailProcessorFactory.create_default_processor()
    processor.process()


if __name__ == "__main__":
    main()
