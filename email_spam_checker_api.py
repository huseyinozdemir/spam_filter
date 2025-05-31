#!/usr/bin/env python3

import sys
import email
from email.policy import default
import requests
import re
import string
from typing import Protocol, Dict, Any


class EmailReader(Protocol):
    def read(self) -> str: ...


class EmailWriter(Protocol):
    def write(self, content: str) -> None: ...


class TextCleaner(Protocol):
    def clean(self, text: str) -> str: ...


class SpamDetectionAPI(Protocol):
    def is_spam(self, text: str) -> bool: ...


class EmailParser(Protocol):
    def parse(self, raw_email: str) -> email.message.EmailMessage: ...


class SubjectModifier(Protocol):
    def modify_if_spam(self, msg: email.message.EmailMessage, is_spam: bool) -> None: ...


class StdinEmailReader:    
    def read(self) -> str:
        return sys.stdin.read()


class StdoutEmailWriter:    
    def write(self, content: str) -> None:
        sys.stdout.write(content)


class SpamTextCleaner:    
    def clean(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'https?://\S+', '[LINK]', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return text


class EmailMessageParser:    
    def parse(self, raw_email: str) -> email.message.EmailMessage:
        return email.message_from_string(raw_email, policy=default)


class SpamSubjectModifier:    
    def modify_if_spam(self, msg: email.message.EmailMessage, is_spam: bool) -> None:
        if is_spam:
            original_subject = msg["Subject"] or ""
            if "---SPAM---" not in original_subject:
                new_subject = f"---SPAM--- {original_subject}"
                msg.replace_header("Subject", new_subject)


# DIP - Depend on abstractions, not concretions
class APISpamDetector:    
    def __init__(self, api_url: str, timeout: int = 30):
        self.api_url = api_url
        self.timeout = timeout
    
    def is_spam(self, text: str) -> bool:
        try:
            response = requests.post(
                self.api_url,
                json={"text": text},
                timeout=self.timeout
            )
            
            if response.ok:
                result = response.json()
                # Flexible response handling
                return result.get("spam", False) or result.get("prediction", 0) == 1
            return False
        except Exception:
            # Fail safe - treat as not spam if API fails
            return False


class EmailSpamProcessorConfig:
    """Configuration for email processor"""
    
    def __init__(self, api_url: str = "http://127.0.0.1:5050/predict", api_timeout: int = 30):
        self.api_url = api_url
        self.api_timeout = api_timeout


class EmailSpamProcessor:
    """High-level email processing logic"""
    
    def __init__(
        self,
        email_reader: EmailReader,
        email_writer: EmailWriter,
        text_cleaner: TextCleaner,
        spam_detector: SpamDetectionAPI,
        email_parser: EmailParser,
        subject_modifier: SubjectModifier
    ):
        self.email_reader = email_reader
        self.email_writer = email_writer
        self.text_cleaner = text_cleaner
        self.spam_detector = spam_detector
        self.email_parser = email_parser
        self.subject_modifier = subject_modifier
    
    def process(self) -> None:
        """OCP - Open for extension, closed for modification"""
        # Read raw email
        raw_email = self.email_reader.read()
        
        # Clean text before API call (consistency with local processing)
        cleaned_text = self.text_cleaner.clean(raw_email)
        
        # Check if spam via API
        is_spam = self.spam_detector.is_spam(cleaned_text)
        
        if is_spam:
            # Parse email for modification
            msg = self.email_parser.parse(raw_email)
            
            # Modify subject
            self.subject_modifier.modify_if_spam(msg, is_spam)
            
            # Write modified email
            self.email_writer.write(msg.as_string())
        else:
            # Write original email unchanged
            self.email_writer.write(raw_email)


# Factory pattern for dependency injection
class EmailProcessorFactory:
    """Factory for creating processor with all dependencies"""
    
    @staticmethod
    def create_default_processor(config: EmailSpamProcessorConfig = None) -> EmailSpamProcessor:
        if config is None:
            config = EmailSpamProcessorConfig()
        
        return EmailSpamProcessor(
            email_reader=StdinEmailReader(),
            email_writer=StdoutEmailWriter(),
            text_cleaner=SpamTextCleaner(),
            spam_detector=APISpamDetector(config.api_url, config.api_timeout),
            email_parser=EmailMessageParser(),
            subject_modifier=SpamSubjectModifier()
        )
    
    @staticmethod
    def create_with_custom_api(api_url: str, timeout: int = 30) -> EmailSpamProcessor:
        config = EmailSpamProcessorConfig(api_url, timeout)
        return EmailProcessorFactory.create_default_processor(config)


def main():
    """Main function with configurable API endpoint"""
    # Easy to configure different API endpoints
    processor = EmailProcessorFactory.create_default_processor()
    
    # Or use custom endpoint:
    # processor = EmailProcessorFactory.create_with_custom_api("http://api.example.com/spam-check")
    
    processor.process()


if __name__ == "__main__":
    main() 