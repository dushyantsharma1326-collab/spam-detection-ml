import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------
# LOAD MACHINE LEARNING MODEL
# -----------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error("Model files could not be loaded.")
    st.write("Make sure these files exist in your repository:")
    st.code("spam_model.pkl\nvectorizer.pkl")
    st.stop()

# -----------------------------------------
# SESSION STATE
# -----------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "total_scanned" not in st.session_state:
    st.session_state.total_scanned = 0

if "spam_detected" not in st.session_state:
    st.session_state.spam_detected = 0

# -----------------------------------------
# CUSTOM DARK THEME
# -----------------------------------------

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
    text-align: center;
}

.subtitle {
    color: #94A3B8;
    font-size: 17px;
    text-align: center;
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
    width: 100%;
}

div.stButton > button:hover {
    background-color: #059669;
    color: white;
}

.stTextArea textarea {
    color: #111827 !important;
    background-color: #FFFFFF !important;
    caret-color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

.stTextArea textarea::placeholder {
    color: #777777 !important;
    -webkit-text-fill-color: #777777 !important;
}

[data-testid="stMetric"] {
    background-color: #172338;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #334155;
}

h1, h2, h3 {
    color: #F8FAFC;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------

with st.sidebar:

    st.title("🛡️ SpamGuard AI")

    st.caption("Intelligent Spam Detection System")

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📜 History", "📊 Analytics"]
    )

    st.divider()

    st.info("Powered by Machine Learning")

    st.caption("SpamGuard AI | Version 1.0")

# -----------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------

def predict_spam(message):

    transformed_message = vectorizer.transform([message])

    prediction = model.predict(transformed_message)[0]

    probability = model.predict_proba(transformed_message)[0]

    classes = list(model.classes_)

    # Find the spam class
    spam_index = None

    for i, label in enumerate(classes):
        if str(label).strip().lower() == "spam":
            spam_index = i
            break

    if spam_index is None:
        for i, label in enumerate(classes):
            if str(label).strip() == "1":
                spam_index = i
                break

    if spam_index is None:
        spam_index = 1 if len(classes) > 1 else 0

    spam_probability = float(probability[spam_index]) * 100

    if str(prediction).strip().lower() == "spam":
        result = "SPAM"
    elif str(prediction).strip().lower() == "ham":
        result = "NOT SPAM"
    elif str(prediction).strip() == "1":
        result = "SPAM"
    else:
        result = "NOT SPAM"

    return result, spam_probability

# -----------------------------------------
# DASHBOARD PAGE
# -----------------------------------------

if page == "🏠 Dashboard":

    st.markdown(
        '<div class="main-title">🛡️ SpamGuard AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Intelligent Spam Message Detection System</div>',
        unsafe_allow_html=True
    )

    st.write("")

    st.header("🔎 Analyze Your Message")

    message = st.text_area(
        "Enter your SMS or message",
        placeholder="Type your message here...",
        height=150,
        key="message_input"
    )

    if st.button("🚀 Analyze Message", use_container_width=True):

        if not message.strip():

            st.warning("Please enter a message first.")

        else:

            result, spam_probability = predict_spam(message)

            st.session_state.total_scanned += 1

            if result == "SPAM":
                st.session_state.spam_detected += 1

            st.session_state.history.append({
                "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                "Message": message,
                "Result": result,
                "Spam Probability": round(spam_probability, 2)
            })

            st.session_state.last_result = result
            st.session_state.last_probability = spam_probability

    # Show latest prediction

    if "last_result" in st.session_state:

        st.divider()

        st.subheader("🎯 Detection Result")

        result = st.session_state.last_result
        probability = st.session_state.last_probability

        if result == "SPAM":

            st.error("🚨 This message is SPAM!")

        else:

            st.success("✅ This message is NOT SPAM!")

        st.metric(
            "Spam Probability",
            f"{probability:.2f}%"
        )

        st.progress(int(probability))

        st.caption(
            "Probability represents the model's estimated confidence "
            "for the spam class."
        )

    # Quick statistics

    st.divider()

    st.header("📊 Quick Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Scanned",
        st.session_state.total_scanned
    )

    col2.metric(
        "Spam Detected",
        st.session_state.spam_detected
    )

    col3.metric(
        "Safe Messages",
        st.session_state.total_scanned
        - st.session_state.spam_detected
    )

# -----------------------------------------
# HISTORY PAGE
# -----------------------------------------

elif page == "📜 History":

    st.title("📜 Detection History")

    st.write("All messages analyzed during this session.")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History CSV",
            data=csv,
            file_name="spamguard_history.csv",
            mime="text/csv"
        )

        if st.button("🗑️ Clear History"):

            st.session_state.history = []
            st.session_state.total_scanned = 0
            st.session_state.spam_detected = 0

            if "last_result" in st.session_state:
                del st.session_state.last_result

            if "last_probability" in st.session_state:
                del st.session_state.last_probability

            st.rerun()

    else:

        st.info("No detection history available yet.")

# -----------------------------------------
# ANALYTICS PAGE
# -----------------------------------------

elif page == "📊 Analytics":

    st.title("📊 SpamGuard Analytics")

    if st.session_state.history:

        df = pd.DataFrame(st.session_state.history)

        total = len(df)

        spam_count = len(
            df[df["Result"] == "SPAM"]
        )

        safe_count = len(
            df[df["Result"] == "NOT SPAM"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Messages", total)

        col2.metric("Spam Messages", spam_count)

        col3.metric("Safe Messages", safe_count)

        st.divider()

        st.subheader("📈 Spam vs Safe Messages")

        chart_data = pd.DataFrame({
            "Category": ["Spam", "Safe"],
            "Messages": [spam_count, safe_count]
        })

        st.bar_chart(
            chart_data,
            x="Category",
            y="Messages"
        )

        st.divider()

        st.subheader("📊 Spam Probability Distribution")

        st.bar_chart(
            df["Spam Probability"]
        )

        st.divider()

        st.subheader("📋 Detailed Analytics")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Analyze some messages on the Dashboard "
            "to generate analytics."
        )

# -----------------------------------------
# FOOTER
# -----------------------------------------

st.divider()

st.markdown(
    "<center>🛡️ SpamGuard AI | Built with Python, "
    "Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)
