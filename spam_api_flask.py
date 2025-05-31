from flask import Flask, request, jsonify
import joblib
import re
import string

app = Flask(__name__)

model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+', '[LINK]', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    cleaned = clean_text(data["text"])
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return jsonify({"prediction": int(prediction),
                    "label": "Spam" if prediction else "Ham"})


if __name__ == "__main__":
    app.run(port=5050)
