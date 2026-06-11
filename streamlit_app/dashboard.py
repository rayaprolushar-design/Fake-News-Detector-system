import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")

# ── Premium UI Styling ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@500;600;700;800;900&display=swap');

/* Apply modern fonts & globals */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.5px;
    font-weight: 700;
}

/* App Background override targeting stAppViewContainer and .stApp */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background-color: #03000a !important;
    background-image: 
        radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.22) 0px, transparent 55%),
        radial-gradient(at 100% 0%, rgba(236, 72, 153, 0.22) 0px, transparent 55%),
        radial-gradient(at 50% 100%, rgba(168, 85, 247, 0.18) 0px, transparent 50%),
        radial-gradient(at 80% 90%, rgba(6, 182, 212, 0.15) 0px, transparent 50%) !important;
    background-attachment: fixed !important;
    color: #e2e8f0 !important;
}

/* Glassmorphism for content container with gradient border */
.main .block-container {
    padding: 3rem 4rem 4rem 4rem !important;
    border-radius: 24px !important;
    background: rgba(13, 10, 28, 0.65) !important;
    box-shadow: 
        0 25px 60px rgba(0, 0, 0, 0.5), 
        0 0 40px rgba(139, 92, 246, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(25px) !important;
    -webkit-backdrop-filter: blur(25px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    margin-top: 3rem !important;
    margin-bottom: 3.5rem !important;
    position: relative !important;
    overflow: hidden !important;
}

/* Neon glow line at the top of the container */
.main .block-container::before {
    content: '' !important;
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    height: 4px !important;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899, #00f2fe, #6366f1) !important;
    background-size: 200% 100% !important;
    animation: animateGlow 4s linear infinite !important;
}

@keyframes animateGlow {
    0% { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* Headers */
h1 {
    background: linear-gradient(45deg, #00f2fe, #4facfe, #8b5cf6, #ec4899);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    animation: gradientShift 6s ease infinite;
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
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

if not feedback_provided.empty:
    accuracy = ((len(feedback_provided) - len(incorrect_preds)) / len(feedback_provided)) * 100
    accuracy_text = f"{accuracy:.1f}%"
else:
    accuracy_text = "No Feedback Yet"

st.markdown(f"""
<div style="display: flex; gap: 20px; justify-content: space-between; flex-wrap: wrap; margin-bottom: 30px;">
    <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(6, 182, 212, 0.03) 100%); border-left: 5px solid #06b6d4; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div style="font-size: 0.9rem; color: #a5f3fc; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">📈 Total Predictions</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #06b6d4; margin-top: 10px;">{total_predictions}</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Total news predictions logged</div>
    </div>
    <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(236, 72, 153, 0.03) 100%); border-left: 5px solid #ec4899; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div style="font-size: 0.9rem; color: #fbcfe8; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">🎯 Average Confidence</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #ec4899; margin-top: 10px;">{avg_confidence:.1f}%</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Mean probability score</div>
    </div>
    <div style="flex: 1; min-width: 200px; background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.03) 100%); border-left: 5px solid #10b981; padding: 20px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.3); border-top: 1px solid rgba(255,255,255,0.08); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);">
        <div style="font-size: 0.9rem; color: #a7f3d0; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">✅ User-Verified Accuracy</div>
        <div style="font-size: 2.5rem; font-weight: 800; color: #10b981; margin-top: 10px;">{accuracy_text}</div>
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 5px;">Based on crowd feedback</div>
    </div>
</div>
""", unsafe_allow_html=True)

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
