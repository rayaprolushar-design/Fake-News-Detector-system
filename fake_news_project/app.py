import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import csv
import datetime
import pandas as pd
import streamlit as st
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from url_scraper import scrape_article
from batch_predictor import run_batch_prediction

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector (Pro)",
    page_icon="🔍",
    layout="centered"
)

# ── Premium UI Styling ───────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Outfit:wght@500;700&display=swap');

/* Apply modern fonts */
html, body, [class*="st-"] {
    font-family: 'Inter', sans-serif;
}
h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.5px;
}

/* App Background with subtle mesh gradient */
.stApp {
    background-color: #0b0f19;
    background-image: 
        radial-gradient(at 0% 0%, hsla(253,16%,7%,1) 0, transparent 50%), 
        radial-gradient(at 50% 0%, hsla(225,39%,30%,0.2) 0, transparent 50%), 
        radial-gradient(at 100% 0%, hsla(339,49%,30%,0.2) 0, transparent 50%);
    background-attachment: fixed;
    color: #e2e8f0;
}

/* Glassmorphism for main content area */
.block-container {
    padding: 3rem 4rem 4rem 4rem !important;
    border-radius: 20px;
    background: rgba(15, 23, 42, 0.4);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 3rem;
    margin-bottom: 3rem;
}

/* Beautiful Inputs */
.stTextArea textarea, .stTextInput input, .stFileUploader {
    background-color: rgba(15, 23, 42, 0.6) !important;
    border: 1px solid rgba(99, 102, 241, 0.3) !important;
    color: #f8fafc !important;
    border-radius: 12px !important;
    padding: 15px !important;
    transition: all 0.3s ease;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2) !important;
}

/* Animated Premium Buttons */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    color: white !important;
    border: none !important;
    border-radius: 12px;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
}
.stButton > button:active {
    transform: translateY(1px);
}

/* Custom Radio Buttons container */
div[role="radiogroup"] {
    background: rgba(15, 23, 42, 0.5);
    padding: 10px 15px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* Metrics and Alerts */
.stAlert {
    border-radius: 12px;
    border: None;
}

/* Headers */
h1 {
    background: -webkit-linear-gradient(45deg, #6366f1, #d946ef);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
}

/* Progress bar modernizing */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #4f46e5, #d946ef);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "bert_model")
LOG_FILE = os.path.join(BASE_DIR, "prediction_log.csv")

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

# ── Logging Setup ───────────────────────────────────
def log_prediction(input_text, label, confidence):
    file_exists = os.path.exists(LOG_FILE)
    df_len = 0
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['ID', 'Timestamp', 'Input_Text', 'Label', 'Confidence', 'User_Feedback'])
        else:
            df_len = len(pd.read_csv(LOG_FILE))
            
        row_id = df_len + 1
        writer.writerow([
            row_id,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text[:300],  # truncate to save space
            label,
            confidence,
            "" # empty feedback initially
        ])
    return row_id

def update_feedback(row_id, feedback):
    if not os.path.exists(LOG_FILE): return
    df = pd.read_csv(LOG_FILE)
    if row_id <= len(df):
        df.loc[df['ID'] == row_id, 'User_Feedback'] = feedback
        df.to_csv(LOG_FILE, index=False)


# ── Prediction Logic ────────────────────────────────
def predict_single(text):
    if len(text.split()) < 6:
        return "UNCERTAIN", 0.0, "Too short — add more text for a reliable result."
    if model is None:
        return "MODEL MISSING", 0.0, "DistilBERT model not found."

    inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    
    fake_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100

    if real_prob >= 75: label, note = "REAL", "High confidence — DistilBERT identifies this as genuine."
    elif fake_prob >= 75: label, note = "FAKE", "High confidence — DistilBERT detects signs of misinformation."
    elif real_prob >= 60: label, note = "LIKELY REAL", "Leaning real, but verify with a trusted source."
    elif fake_prob >= 60: label, note = "LIKELY FAKE", "Leaning fake, but check before sharing."
    else: label, note = "UNCERTAIN", "The model isn't confident — please verify manually."

    return label, round(max(real_prob, fake_prob), 1), note

def _show_result(label, confidence, note, raw_text):
    st.divider()
    if label == "REAL": st.success(f"### ✅ {label}")
    elif label == "FAKE": st.error(f"### ❌ {label}")
    elif label in ("LIKELY REAL", "LIKELY FAKE"): st.warning(f"### ⚠️ {label}")
    else: st.info(f"### ❓ {label}")

    st.markdown(note)
    if confidence > 0:
        st.markdown(f"**Confidence: {confidence}%**")
        st.progress(int(confidence))
        
    st.caption("Always verify important news with trusted sources.")
    
    # Logging and Feedback mechanism
    if "current_log_id" not in st.session_state or st.session_state.get('last_text') != raw_text:
        st.session_state.current_log_id = log_prediction(raw_text, label, confidence)
        st.session_state.last_text = raw_text
        st.session_state.feedback_given = False

    if not st.session_state.feedback_given:
        st.write("Was this prediction correct?")
        c1, c2, c3 = st.columns([1,1,3])
        with c1:
            if st.button("👍 Yes"):
                update_feedback(st.session_state.current_log_id, "Correct")
                st.session_state.feedback_given = True
                st.rerun()
        with c2:
            if st.button("👎 No"):
                update_feedback(st.session_state.current_log_id, "Incorrect")
                st.session_state.feedback_given = True
                st.rerun()
    else:
        st.success("Thanks for improving the model! Check the Dashboard.")

# ── Header ───────────────────────────────────────────
st.title("🔍 Fake News Detector (Pro)")
st.markdown("Powered by DistilBERT. Analyze raw text, scrape live URLs, or upload CSV batches!")
st.divider()

# ── 3-Mode Selector ──────────────────────────────────
mode = st.radio("Select Input Mode:", ["📝 Paste Text", "🌐 Scrape URL", "📂 Batch CSV Upload"], horizontal=True)

if mode == "📝 Paste Text":
    user_input = st.text_area("Enter news text here", height=160)
    if st.button("🔎 Analyse", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analysing semantic patterns..."):
                label, conf, note = predict_single(user_input)
            _show_result(label, conf, note, user_input)

elif mode == "🌐 Scrape URL":
    url_input = st.text_input("Paste a news article URL:")
    if st.button("🌐 Scrape & Analyse", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching and cleaning article..."):
                article = scrape_article(url_input)
            
            if article['error']:
                st.error(article['error'])
            else:
                st.info(f"**Found Article:** {article['title']}")
                with st.spinner("Analysing semantics..."):
                    label, conf, note = predict_single(article['text'])
                _show_result(label, conf, note, article['text'])

elif mode == "📂 Batch CSV Upload":
    st.info("Upload a CSV file containing headlines to analyze them all at once. The column should be named 'text' or 'headline'.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and st.button("🚀 Run Batch Predictor", use_container_width=True):
        with st.spinner("Processing batch with PyTorch DataLoader..."):
            df_upload = pd.read_csv(uploaded_file)
            col_name = 'text' if 'text' in df_upload.columns else ('headline' if 'headline' in df_upload.columns else df_upload.columns[0])
            
            texts = df_upload[col_name].dropna().astype(str).tolist()
            
            # Predict
            df_results = run_batch_prediction(texts, model, tokenizer, device, batch_size=32)
            
            st.success("Batch Prediction Complete!")
            st.dataframe(df_results, use_container_width=True)
            
            csv_export = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_export,
                file_name='batch_predictions_results.csv',
                mime='text/csv',
            )

# ── Footer ───────────────────────────────────────────
st.divider()
st.caption("Powered by DistilBERT · 99.61% accuracy")
