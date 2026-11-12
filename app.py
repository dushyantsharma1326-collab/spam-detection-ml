import streamlit as st
import joblib

# Load saved model and vectorizer
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="📩"
)

st.title("📩 AI Spam Message Detector")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# User input
message = st.text_area("Enter your message:")

if st.button("Check Message"):
    if message.strip():
        message_vector = vectorizer.transform([message])
        prediction = model.predict(message_vector)[0]

        if prediction == 1:
            st.error("🚨 SPAM MESSAGE")
        else:
            st.success("✅ NOT SPAM MESSAGE")
    else:
        st.warning("Please enter a message.")
