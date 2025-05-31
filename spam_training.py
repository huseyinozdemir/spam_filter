import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text):
    # 1. lowercase
    text = text.lower()
    text = re.sub(r'https?://\S+', '[LINK]', text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Remove numbers
    text = re.sub(r'\d+', '', text)
    # 5. remove stopwords
    # stop_words = set(stopwords.words('turkish'))
    # text = ' '.join(
    #     [word for word in text.split() if word not in stop_words])
    return text

# for testing without the dataset csv
# data = {
#    "text": [
#        "Congratulations! You have won a prize. Click here: [LINK]",  # spam
#        "Clicl here to claim your prize: [LINK]",  # spam
#        "You win: [LINK]",  # spam
#        "Meeting tomorrow at 10'o clock.",  # not spam
#        "See you tomorrow.",  # not spam
#        "Are you attending the meeting today?",  # not spam
#    ],
#    "label": [1, 1, 1, 0, 0, 0]
# }
# df = pd.DataFrame(data)


df = pd.read_csv("spam_dataset.csv")
df['clean_text'] = df['text'].apply(clean_text)

# Convert text to numerical features
# vectorizer = CountVectorizer()

# TfidfVectorizer is better for text classification because
# it considers the importance of words
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

# vectorizer = LogisticRegression()

X = vectorizer.fit_transform(df['clean_text'])
y = df['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create and train the model for small dataset
# model = SVC()

# LogisticRegression is better for text classification because
# it considers the importance of words
# This gives more weight to the minority class (ham message) and
# learns balanced decisions.
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Doğruluk: {accuracy}')

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

new_text = ["Congratulations! You have won a prize. Click here"]
new_text_vectorized = vectorizer.transform([clean_text(new_text[0])])
prediction = model.predict(new_text_vectorized)
print(f'Prediction: {"Spam" if prediction[0] == 1 else "Ham"}')
