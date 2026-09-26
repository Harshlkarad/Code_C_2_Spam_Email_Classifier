import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = pd.read_csv("dataset/spam.csv", encoding="latin-1")

# Keep required columns
data = data.iloc[:, :2]
data.columns = ["label", "message"]

# Convert labels
data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# Remove missing values
data.dropna(inplace=True)

print("Dataset loaded successfully!")
print(f"Total messages: {len(data)}")

# Features and target
X = data["message"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediction
y_pred = model.predict(X_test_tfidf)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("SPAM EMAIL CLASSIFIER")
print("=" * 50)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Not Spam", "Spam"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Test custom messages
def predict_message(message):
    message_tfidf = vectorizer.transform([message])
    prediction = model.predict(message_tfidf)[0]

    if prediction == 1:
        return "SPAM"
    else:
        return "NOT SPAM"


print("\n" + "=" * 50)
print("CUSTOM MESSAGE TEST")
print("=" * 50)

test_messages = [
    "Congratulations! You have won a free iPhone. Click now!",
    "Hi, please send me the project report by evening.",
    "URGENT! You have won a cash prize. Claim now!",
    "Can you call me when you are free?"
]

for message in test_messages:
    result = predict_message(message)

    print(f"\nMessage: {message}")
    print(f"Prediction: {result}")