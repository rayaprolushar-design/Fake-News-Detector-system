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
    # Guard: too short to be reliable
    if len(text.split()) < 6:
        return "UNCERTAIN", 0.0, "Too short — add more text for a reliable result."

    cleaned    = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    proba      = model.predict_proba(vectorized)[0]

    real_prob = proba[1] * 100
    fake_prob = proba[0] * 100

    # Confidence threshold zones
    if real_prob >= 75:
        label, note = "REAL", "High confidence — this looks like real news."
    elif fake_prob >= 75:
        label, note = "FAKE", "High confidence — this shows signs of misinformation."
    elif real_prob >= 60:
        label, note = "LIKELY REAL", "Leaning real, but verify with a trusted source."
    elif fake_prob >= 60:
        label, note = "LIKELY FAKE", "Leaning fake, but check before sharing."
    else:
        label, note = "UNCERTAIN", "The model isn't confident — please verify manually."

    confidence = round(max(real_prob, fake_prob), 1)
    return label, confidence, note

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
            label, confidence, note = predict(user_input)

        st.divider()

        # Colour the result based on verdict
        if label == "REAL":
            st.success(f"### ✅ {label}")
        elif label == "FAKE":
            st.error(f"### ❌ {label}")
        elif label in ("LIKELY REAL", "LIKELY FAKE"):
            st.warning(f"### ⚠️ {label}")
        else:
            st.info(f"### ❓ {label}")

        st.markdown(note)

        if confidence > 0:
            st.markdown(f"**Confidence: {confidence}%**")
            st.progress(int(confidence))

        st.caption("Always verify important news with trusted sources.")

# ── Footer ───────────────────────────────────────────
st.divider()
st.caption("""
Built with Python · scikit-learn · Streamlit  
Trained on 44,919 articles · 98.74% accuracy
""")
