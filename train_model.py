
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# 2. Separate messages and labels
X = data["message"]
y = data["label"].map({"ham": 0, "spam": 1})

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Convert text into numbers
vectorizer = CountVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Create and train model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# 6. Make predictions
predictions = model.predict(X_test_vectorized)

# 7. Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("Model Training Completed!")
print("Total messages:", len(data))
print("Training messages:", len(X_train))
print("Testing messages:", len(X_test))

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

import joblib

# Save trained model and vectorizer
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model saved successfully!")


while True:

    message = input("\nEnter your message: ")

    if message.lower() == "exit":
        print("Thank you for using AI Spam Detector!")
        break

    # Convert message into numbers
    message_vector = vectorizer.transform([message])

    # Predict
    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        print("🚨 SPAM MESSAGE")
    else:
        print("✅ NOT A SPAM MESSAGE")

        