import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analytics Dashboard", page_icon="📊", layout="wide")
st.title("📊 Model Performance & Analytics Dashboard")

LOG_FILE = "prediction_log.csv"

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
