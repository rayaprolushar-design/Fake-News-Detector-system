import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# ── Premium UI Styling ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.5px;
}

.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
    background-attachment: fixed;
    color: #e2e8f0;
}

.stMetric {
    background: rgba(15, 23, 42, 0.6);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.block-container {
    padding: 3rem 4rem 4rem 4rem !important;
}

h1 {
    background: -webkit-linear-gradient(45deg, #6366f1, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("📊 Model Performance & Analytics Dashboard")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, "prediction_log.csv")

if not os.path.exists(LOG_FILE):
    st.info("No predictions yet! Use the main app to generate some data.")
    st.stop()

# Load data gracefully
try:
    df = pd.read_csv(LOG_FILE)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
except Exception as e:
    st.error(f"Error reading log file: {e}")
    st.stop()

# ── Metrics ──
total_predictions = len(df)
avg_confidence = df['Confidence'].mean()
feedback_provided = df[df['User_Feedback'].notna()]
incorrect_preds = feedback_provided[feedback_provided['User_Feedback'] == 'Incorrect']

col1, col2, col3 = st.columns(3)
col1.metric("Total Predictions", total_predictions)
col2.metric("Average Confidence", f"{avg_confidence:.1f}%")

if not feedback_provided.empty:
    accuracy = ((len(feedback_provided) - len(incorrect_preds)) / len(feedback_provided)) * 100
    col3.metric("User-Verified Accuracy", f"{accuracy:.1f}%")
else:
    col3.metric("User-Verified Accuracy", "No Feedback Yet")

st.divider()

# ── Charts ──
c1, c2 = st.columns(2)

with c1:
    st.subheader("Verdict Distribution")
    dist = df['Label'].value_counts()
    fig, ax = plt.subplots()
    sns.barplot(x=dist.index, y=dist.values, palette='viridis', ax=ax)
    st.pyplot(fig)

with c2:
    st.subheader("Confidence Spread")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Confidence'], bins=10, kde=True, color='#2C7BB6', ax=ax2)
    st.pyplot(fig2)

st.divider()

# ── Wrong Predictions Log ──
st.subheader("🚨 Continuous Improvement: Wrong Predictions Log")
st.markdown("These are predictions the model made that users manually flagged as incorrect:")

if not incorrect_preds.empty:
    st.dataframe(incorrect_preds[['Timestamp', 'Input_Text', 'Label', 'Confidence']], use_container_width=True)
else:
    st.success("No incorrect predictions flagged yet! Outstanding performance.")

st.divider()
st.subheader("Full Prediction History")
st.dataframe(df.sort_values(by="Timestamp", ascending=False), use_container_width=True)
