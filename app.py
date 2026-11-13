
import streamlit as st
import pandas as pd
import joblib
import sqlite3
import plotly.express as px
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. PREMIUM DARK THEME
# ==========================================

st.markdown("""
<style>

.stApp {
    background-color: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

h1, h2, h3 {
    color: #f1f5f9;
}

p, label {
    color: #cbd5e1;
}

/* PREMIUM STATISTIC CARDS */

.metric-card {
    background: linear-gradient(135deg, #1e293b, #312e81);
    padding: 24px;
    border-radius: 15px;
    border: 1px solid #475569;
    box-shadow: 0 5px 20px rgba(0,0,0,0.25);
    transition: 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    border-color: #818cf8;
}

.metric-card h1 {
    color: #ffffff;
    font-size: 38px;
    margin: 8px 0;
}

.metric-card h4 {
    color: #c7d2fe;
}

.metric-card p {
    color: #94a3b8;
}

/* MESSAGE INPUT */

.stTextArea textarea {
    background-color: #1e293b;
    color: white;
    border: 1px solid #475569;
    border-radius: 12px;
}

/* BUTTONS */

.stButton button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton button:hover {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: 1px solid #a5b4fc;
}

/* RESULT CARDS */

.result-spam {
    background-color: #451a1a;
    border: 1px solid #ef4444;
    padding: 22px;
    border-radius: 14px;
}

.result-safe {
    background-color: #123524;
    border: 1px solid #22c55e;
    padding: 22px;
    border-radius: 14px;
}

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD MACHINE LEARNING MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    return model, vectorizer


try:

    model, vectorizer = load_model()

except Exception as e:

    st.error(f"Model loading error: {e}")
    st.stop()


# ==========================================
# 4. DATABASE CONFIGURATION
# ==========================================

def initialize_database():

    conn = sqlite3.connect("spamguard.db")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            message TEXT,
            result TEXT,
            probability REAL
        )
    """)

    conn.commit()
    conn.close()


initialize_database()


# ==========================================
# 5. SAVE DETECTION HISTORY
# ==========================================

def save_history(message, result, probability):

    conn = sqlite3.connect("spamguard.db")

    conn.execute("""
        INSERT INTO history
        (timestamp, message, result, probability)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        message,
        result,
        probability
    ))

    conn.commit()
    conn.close()


# ==========================================
# 6. GET HISTORY
# ==========================================

def get_history():

    conn = sqlite3.connect("spamguard.db")

    df = pd.read_sql_query(
        "SELECT * FROM history ORDER BY id DESC",
        conn
    )

    conn.close()

    return df


# ==========================================
# 7. CLEAR HISTORY
# ==========================================

def clear_history():

    conn = sqlite3.connect("spamguard.db")

    conn.execute("DELETE FROM history")

    conn.commit()
    conn.close()


# ==========================================
# 8. SPAM DETECTION FUNCTION
# ==========================================

def predict_spam(message):

    message_vector = vectorizer.transform([message])

    prediction = model.predict(message_vector)[0]

    label = str(prediction).strip().lower()

    # Determine prediction result

    if label in ["spam", "1", "1.0"]:

        result = "Spam"

    elif label in ["ham", "safe", "not spam", "0", "0.0"]:

        result = "Safe"

    else:

        raise ValueError(
            f"Unknown model prediction label: {prediction}"
        )

    # Calculate probability

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(message_vector)[0]

        classes = list(model.classes_)

        spam_index = None

        for i, class_label in enumerate(classes):

            class_name = str(class_label).strip().lower()

            if class_name in ["spam", "1", "1.0"]:

                spam_index = i
                break

        if spam_index is not None:

            spam_probability = float(
                probabilities[spam_index]
            ) * 100

        else:

            spam_probability = (
                100.0 if result == "Spam" else 0.0
            )

    else:

        spam_probability = (
            100.0 if result == "Spam" else 0.0
        )

    return result, spam_probability


# ==========================================
# 9. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🛡️ SpamGuard AI")

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📜 History",
        "📊 Analytics"
    ]
)

st.sidebar.markdown("---")

st.sidebar.info(
    "AI-powered SMS spam detection system."
)

st.sidebar.caption("Version 2.0")


# ==========================================
# 10. DASHBOARD PAGE
# ==========================================

if page == "🏠 Dashboard":

    st.title("🛡️ SpamGuard AI")

    st.caption(
        "YOUR INTELLIGENT MESSAGE SECURITY CENTER"
    )

    st.markdown("---")

    # GET STATISTICS

    df = get_history()

    total = len(df)

    spam_count = len(
        df[df["result"] == "Spam"]
    )

    safe_count = len(
        df[df["result"] == "Safe"]
    )

    # PREMIUM STATISTIC CARDS

    st.subheader("📊 Security Overview")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="metric-card">
            <h4>📩 Total Messages</h4>
            <h1>{total}</h1>
            <p>Messages Analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="metric-card">
            <h4>🚨 Spam Detected</h4>
            <h1>{spam_count}</h1>
            <p>Threats Identified</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        st.markdown(f"""
        <div class="metric-card">
            <h4>✅ Safe Messages</h4>
            <h1>{safe_count}</h1>
            <p>Legitimate Messages</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # MESSAGE ANALYZER

    st.subheader("🔍 AI Message Analyzer")

    st.write(
        "Check any SMS message using our trained Machine Learning model."
    )

    message = st.text_area(
        "Enter your message",
        placeholder="Paste your SMS message here...",
        height=160,
        key="premium_message"
    )

    analyze = st.button(
        "🚀 ANALYZE MESSAGE",
        use_container_width=True
    )

    if analyze:

        if not message.strip():

            st.warning(
                "Please enter a message first."
            )

        else:

            try:

                result, probability = predict_spam(message)

                save_history(
                    message,
                    result,
                    probability
                )

                st.session_state["last_result"] = result

                st.session_state["last_probability"] = probability

                st.session_state["last_message"] = message

                st.success(
                    "Message analyzed successfully!"
                )

            except Exception as e:

                st.error(
                    f"Prediction error: {e}"
                )

    # DISPLAY LATEST RESULT

    if "last_result" in st.session_state:

        result = st.session_state["last_result"]

        probability = st.session_state["last_probability"]

        st.markdown("---")

        st.subheader("🎯 Latest Analysis Result")

        col1, col2 = st.columns(2)

        with col1:

            if result == "Spam":

                st.markdown("""
                <div class="result-spam">
                    <h2>🚨 SPAM MESSAGE</h2>
                    <p>This message has been identified as spam.</p>
                </div>
                """, unsafe_allow_html=True)

            else:

                st.markdown("""
                <div class="result-safe">
                    <h2>✅ SAFE MESSAGE</h2>
                    <p>This message appears legitimate.</p>
                </div>
                """, unsafe_allow_html=True)

        with col2:

            st.metric(
                "Spam Probability",
                f"{probability:.2f}%"
            )

            st.progress(
                min(max(int(probability), 0), 100)
            )

        st.caption(
            "Probability represents the model's estimated likelihood "
            "that the message is spam."
        )

        st.caption(
            "Machine learning predictions are estimates, not guarantees."
        )

    st.markdown("---")

    # RECENT ACTIVITY

    st.subheader("🕒 Recent Detection Activity")

    recent_df = get_history()

    if recent_df.empty:

        st.info(
            "No messages analyzed yet."
        )

    else:

        st.dataframe(
            recent_df.head(5),
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 11. HISTORY PAGE
# ==========================================

elif page == "📜 History":

    st.title("📜 Detection History")

    st.caption(
        "View all previously analyzed messages."
    )

    df = get_history()

    if df.empty:

        st.info(
            "No detection history available."
        )

    else:

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Records",
                len(df)
            )

        with col2:

            if st.button("🗑️ Clear History"):

                clear_history()

                st.success(
                    "History cleared successfully!"
                )

                st.rerun()

        # CSV DOWNLOAD

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download History CSV",
            data=csv,
            file_name="spamguard_history.csv",
            mime="text/csv"
        )


# ==========================================
# 12. ANALYTICS PAGE
# ==========================================

elif page == "📊 Analytics":

    st.title("📊 Analytics Dashboard")

    st.caption(
        "Interactive insights from SpamGuard AI"
    )

    df = get_history()

    if df.empty:

        st.info(
            "Analyze some messages to see analytics."
        )

    else:

        total = len(df)

        spam_count = len(
            df[df["result"] == "Spam"]
        )

        safe_count = len(
            df[df["result"] == "Safe"]
        )

        spam_percentage = (
            spam_count / total * 100
        )

        safe_percentage = (
            safe_count / total * 100
        )

        # METRIC CARDS

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Messages",
            total
        )

        col2.metric(
            "🚨 Spam Messages",
            spam_count
        )

        col3.metric(
            "✅ Safe Messages",
            safe_count
        )

        st.markdown("---")

        # DONUT CHART

        st.subheader("🍩 Message Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Spam", "Safe"],
            "Count": [spam_count, safe_count]
        })

        fig = px.pie(
            chart_data,
            names="Category",
            values="Count",
            hole=0.55,
            title="Spam vs Safe Messages",
            color="Category",
            color_discrete_map={
                "Spam": "#ef4444",
                "Safe": "#22c55e"
            }
        )

        fig.update_traces(
            textinfo="label+percent",
            hovertemplate=(
                "%{label}: %{value}<extra></extra>"
            )
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            font_color="white"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        # BAR CHART

        st.subheader(
            "📈 Spam vs Safe Comparison"
        )

        bar_fig = px.bar(
            chart_data,
            x="Category",
            y="Count",
            color="Category",
            text="Count",
            color_discrete_map={
                "Spam": "#ef4444",
                "Safe": "#22c55e"
            }
        )

        bar_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0b1120",
            plot_bgcolor="#0b1120",
            showlegend=False
        )

        st.plotly_chart(
            bar_fig,
            use_container_width=True
        )

        st.markdown("---")

        # SPAM DISTRIBUTION

        st.subheader("🎯 Spam Detection Rate")

        col1, col2 = st.columns(2)

        col1.metric(
            "Spam Percentage",
            f"{spam_percentage:.2f}%"
        )

        col2.metric(
            "Safe Percentage",
            f"{safe_percentage:.2f}%"
        )

        st.progress(
            min(max(int(spam_percentage), 0), 100)
        )

        st.markdown("---")

        # RECENT RECORDS

        st.subheader(
            "🕒 Recent Detection Records"
        )

        st.dataframe(
            df.head(10),
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 13. FOOTER
# ==========================================

st.markdown("---")

st.markdown("""
<div class="footer">
    🛡️ SpamGuard AI | Version 2.0
    <br>
    Powered by Machine Learning
    <br>
    Built with Python, Streamlit and Plotly
</div>
""", unsafe_allow_html=True)

