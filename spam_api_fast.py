from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re
import string

app = FastAPI()

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class Message(BaseModel):
    text: str


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+', '[LINK]', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text


@app.post("/predict")
def predict(msg: Message):
    print(msg)
    cleaned = clean_text(msg.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return {
        "prediction": int(prediction),
        "label": "Spam" if prediction == 1 else "Ham"
    }
