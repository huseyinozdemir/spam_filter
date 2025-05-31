1. **Labels**

   - The `df['label']` column contains only two possible values (e.g., `1` = “spam”, `0` = “ham”).
   - The model decides between these two classes.

2. **Model Selection**

   - A `LogisticRegression(class_weight='balanced')` is used. Logistic regression is typically employed to distinguish between two classes.
   - The `class_weight='balanced'` parameter gives more weight to the minority class (e.g., if there are fewer “spam” examples) to help the model learn balanced decisions.

3. **Prediction Phase**
   - Each call to `model.predict(...)` outputs either `0` or `1` for a new text.
   - The line
     ```python
     print(f'Predict: {"Spam" if prediction[0] == 1 else "Ham"}')
     ```
     interprets those two values as “Spam” or “Ham.”

# Installation

```bash
python3.12 -m venv .venv
pip install -r requirements.txt
```

# Start with Fast Api

```bash
uvicorn spam_api_fast:app --reload --host 0.0.0.0 --port 5050
```

# Start with Flask
```bash
python spam_api_flask.py
```