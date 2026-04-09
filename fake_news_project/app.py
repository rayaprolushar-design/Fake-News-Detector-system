import os
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
