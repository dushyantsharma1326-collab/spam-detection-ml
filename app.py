import joblib

# Load saved AI model
model = joblib.load("spam_model.pkl")

# Load saved vectorizer
vectorizer = joblib.load("vectorizer.pkl")

print("===== AI SPAM DETECTOR =====")

while True:

    message = input("\nEnter your message (or type exit): ")

    if message.lower() == "exit":
        print("Thank you!")
        break

    # Convert message into numbers
    message_vector = vectorizer.transform([message])

    # Predict message
    prediction = model.predict(message_vector)[0]

    if prediction == 1:
        print("🚨 SPAM MESSAGE")
    else:
        print("✅ NOT SPAM MESSAGE")