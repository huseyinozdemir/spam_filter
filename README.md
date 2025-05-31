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

# Training

```bash
python spam_training.py
```

# Start with Fast Api

```bash
uvicorn spam_api_fast:app --reload --host 0.0.0.0 --port 5050
```

# Start with Flask

```bash
python spam_api_flask.py
```

# Test for console

```bash
python test_email_checker.py
```

## Response

```bash
Email Spam Checker Test
==================================================

--------------------------------------------------
Test: Normal Email Test
File: test_email_ham.txt
--------------------------------------------------
Çıktı:
From: ahmet@example.com
To: mehmet@example.com
Subject: =?utf-8?q?Toplant=C4=B1_Hat=C4=B1rlatmas=C4=B1?=
Date: Mon, 15 Jan 2024 10:30:00 +0300
Content-Type: text/plain; charset=UTF-8

Merhaba Mehmet,

Yarın saat 14:00'te konferans salonunda yapılacak olan proje toplantısını hatırlatmak istiyorum.

Toplantıda aşağıdaki konuları ele alacağız:
- Proje ilerlemesi
- Önümüzdeki hafta planları
- Bütçe durumu

Katılım için teşekkürler.

İyi çalışmalar,
Ahmet
✅ Normal email

--------------------------------------------------
Test: Spam Email Test
File: test_email_spam.txt
--------------------------------------------------
Çıktı:
From: winner@lottery.com
To: lucky@example.com
Subject: ---SPAM--- CONGRATULATIONS! You Won $1,000,000!!!
Date: Mon, 15 Jan 2024 09:15:00 +0300
Content-Type: text/plain; charset=UTF-8

CONGRATULATIONS!!!

You have WON $1,000,000 in our INTERNATIONAL LOTTERY!

CLAIM YOUR PRIZE NOW by clicking the link below:
http://fake-lottery-scam.com/claim

This is a LIMITED TIME OFFER! Act NOW or lose your winnings FOREVER!

FREE MONEY! NO CATCH! 100% GUARANTEED!

Click here: http://fake-lottery-scam.com/claim-now

Hurry! This offer expires in 24 hours!
🔴 SPAM detected!
```
