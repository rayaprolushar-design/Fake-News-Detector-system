import os
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "bert_model")

# ── Load model & tokenizer ──────────────────────────

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_DIR):
        return None, None, None
        
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    
    return model, tokenizer, device

model, tokenizer, device = load_models()


def predict(text):
    # Guard: too short to be reliable
    if len(text.split()) < 6:
        return "UNCERTAIN", 0.0, "Too short — add more text for a reliable result."
        
    if model is None:
        return "MODEL MISSING", 0.0, "Please run train_bert.py to generate bert_model/ first."

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    fake_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100

    # Confidence threshold zones
    if real_prob >= 75:
        label, note = "REAL", "High confidence — DistilBERT identifies this as genuine."
    elif fake_prob >= 75:
        label, note = "FAKE", "High confidence — DistilBERT detects signs of misinformation."
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
DistilBERT will analyze the semantic context to determine if it looks real or fake.
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
        with st.spinner("Analysing semantic patterns..."):
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
Powered by DistilBERT · 99.61% accuracy
""")
