import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide"
)

# Load existing ML model
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
}
[data-testid="stSidebar"] {
    background-color: #111827;
}
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #4ADEA5;
}
.subtitle {
    color: #94A3B8;
    font-size: 17px;
}
.result-card {
    padding: 22px;
    border-radius: 14px;
    background: #172338;
    border: 1px solid #334155;
}
div.stButton > button {
    background-color: #10B981;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡️ SpamGuard AI")
    st.caption("Smart Message Protection")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Dashboard", "History", "Analytics"]
    )
    st.divider()
    st.caption("Powered by Machine Learning")

# Dashboard
if page == "Dashboard":

    st.markdown(
        '<div class="main-title">SpamGuard AI</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Intelligent Spam Message Detection System</div>',
        unsafe_allow_html=True
    )

    st.write("")
    st.markdown("### 🔍 Analyze Your Message")

    message = st.text_area(
        "Enter your SMS or message",
        placeholder="Paste your message here...",
        height=150
    )

    if st.button("🚀 Analyze Message", use_container_width=True):

        if message.strip():

            message_vector = vectorizer.transform([message])
            prediction = model.predict(message_vector)[0]

            if prediction == 1:
                result = "SPAM"
            else:
                result = "NOT SPAM"

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(message_vector)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    probability = float(
                        probabilities[classes.index(1)]
                    ) * 100

            st.session_state.history.append({
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M"),
                "Message": message,
                "Result": result,
                "Spam Probability": probability
            })

            st.divider()

            if result == "SPAM":
                st.error("🚨 SPAM MESSAGE DETECTED")
            else:
                st.success("✅ THIS MESSAGE IS NOT SPAM")

            if probability is not None:
                st.metric(
                    "Spam Probability",
                    f"{probability:.2f}%"
                )
            else:
                st.info(
                    "Your model does not support probability estimates."
                )

        else:
            st.warning("Please enter a message first.")

    st.divider()
    st.markdown("### 📊 Quick Statistics")

    total = len(st.session_state.history)
    spam = sum(
        x["Result"] == "SPAM"
        for x in st.session_state.history
    )
    safe = total - spam

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Scanned", total)
    c2.metric("Spam Detected", spam)
    c3.metric("Safe Messages", safe)

# History page
elif page == "History":

    st.title("📜 Prediction History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        if st.button("Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No messages analyzed yet.")

# Analytics page
elif page == "Analytics":

    st.title("📈 Analytics Dashboard")

    total = len(st.session_state.history)
    spam = sum(
        x["Result"] == "SPAM"
        for x in st.session_state.history
    )
    safe = total - spam

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Messages", total)
    c2.metric("Spam Messages", spam)
    c3.metric("Safe Messages", safe)

    st.subheader("Message Distribution")

    if total > 0:
        chart = pd.DataFrame({
            "Category": ["Spam", "Not Spam"],
            "Messages": [spam, safe]
        })
        st.bar_chart(
            chart.set_index("Category")
        )
    else:
        st.info("Analyze messages to generate charts.")
