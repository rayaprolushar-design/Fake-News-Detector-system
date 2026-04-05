import os
import streamlit as st
import pickle
from text_processing import clean_text

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load model & vectorizer ──────────────────────────
@st.cache_resource
def load_models():
    with open(os.path.join(BASE_DIR, 'lr_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_models()

def predict(text):
    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    label      = "REAL" if pred == 1 else "FAKE"
    confidence = round(max(proba) * 100, 1)
    return label, confidence

# ── Header ───────────────────────────────────────────
st.title("🔍 Fake News Detector")
st.markdown("""
Paste a **news headline or article** below.
The model will tell you if it looks real or fake, and how confident it is.
""")
st.divider()

# ── Input ────────────────────────────────────────────
user_input = st.text_area(
    "Enter news text here",
    placeholder="e.g. Scientists confirm the vaccine is 95% effective...",
    height=160
)

col1, col2 = st.columns([2, 5])

with col1:
    analyse = st.button("🔎 Analyse", use_container_width=True)

with col2:
    if st.button("Clear", use_container_width=True):
        st.rerun()

# ── Prediction output ────────────────────────────────
if analyse:
    if not user_input.strip():
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analysing..."):
            label, confidence = predict(user_input)

        st.divider()

        if label == "FAKE":
            st.error(f"### FAKE NEWS DETECTED")
            st.markdown(f"The model is **{confidence}% confident** this is fake.")
        else:
            st.success(f"### LOOKS REAL")
            st.markdown(f"The model is **{confidence}% confident** this is real.")

        # Confidence bar
        st.markdown("**Confidence score**")
        st.progress(int(confidence))

        # Show what the model focused on
        st.divider()
        st.markdown("**What the model saw after cleaning:**")
        cleaned_preview = clean_text(user_input)[:300]
        st.code(cleaned_preview)

# ── Footer ───────────────────────────────────────────
st.divider()
st.caption("""
Built with Python · scikit-learn · Streamlit  
Trained on 44,919 articles · 98.74% accuracy
""")
