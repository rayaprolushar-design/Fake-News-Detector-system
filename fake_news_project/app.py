import os
import sys
import csv
import datetime
import pickle
import pandas as pd
import scipy.sparse
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from text_processing import clean_text
from features import extract_features
from url_scraper import scrape_article
from batch_predictor import run_batch_prediction
from ai_detector import detect_ai_text
from rewrite_guide import get_rewrite_tips
from image_detector import detect_ai_image
from query_search import search_google_news
from factcheck_api import check_google_factcheck, render_factcheck_results

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector (Pro)",
    page_icon="🔍",
    layout="centered"
)

# Prevents Streamlit from rerunning when user types
if 'initialized' not in st.session_state:
    st.session_state.initialized   = True
    st.session_state.last_result   = None
    st.session_state.analysis_count = 0

# ── Premium UI Styling (Updated Ultra-premium targeted CSS) ──
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

/* Restyle top header background to clear */
[data-testid="stHeader"] {
    background-color: transparent !important;
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

/* Beautiful Sidebar Override */
[data-testid="stSidebar"] {
    background-color: #060410 !important;
    border-right: 1px solid rgba(139, 92, 246, 0.15) !important;
}

/* Premium Inputs Targeting baseweb structures */
div[data-baseweb="textarea"] textarea, div[data-baseweb="input"] input {
    background-color: rgba(6, 4, 16, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
    font-size: 1rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
div[data-baseweb="textarea"] textarea:focus, div[data-baseweb="input"] input:focus {
    border-color: #ec4899 !important;
    box-shadow: 0 0 14px rgba(236, 72, 153, 0.3) !important;
    background-color: rgba(6, 4, 16, 0.9) !important;
}

/* Premium Buttons with Glowing Gradient */
[data-testid="stButton"] button {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3) !important;
    width: 100% !important;
    font-size: 1rem !important;
}
[data-testid="stButton"] button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(236, 72, 153, 0.45) !important;
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 50%, #00f2fe 100%) !important;
}
[data-testid="stButton"] button:active {
    transform: translateY(1px) !important;
}
button[key*="clear"] {
    background: rgba(15, 10, 30, 0.5) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    box-shadow: none !important;
}
button[key*="clear"]:hover {
    background: rgba(30, 20, 50, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.15) !important;
    box-shadow: none !important;
}

/* Premium Radio Selector styled as horizontal pill container */
div[data-testid="stRadio"] div[role="radiogroup"] {
    background: rgba(8, 5, 20, 0.55) !important;
    border: 1px solid rgba(139, 92, 246, 0.2) !important;
    border-radius: 16px !important;
    padding: 10px !important;
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 10px !important;
    padding: 6px 14px !important;
    margin: 2px !important;
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    cursor: pointer !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(139, 92, 246, 0.12) !important;
    border-color: rgba(139, 92, 246, 0.4) !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
    border-color: transparent !important;
    box-shadow: 0 4px 15px rgba(236, 72, 153, 0.35) !important;
    color: #ffffff !important;
}

/* Premium Select Box */
div[data-baseweb="select"] > div {
    background-color: rgba(6, 4, 16, 0.8) !important;
    border: 1px solid rgba(139, 92, 246, 0.25) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
}

/* Alerts & Notifications */
.stAlert {
    border-radius: 14px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    background-color: rgba(10, 5, 25, 0.6) !important;
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

/* Progress bar modernizing */
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #00f2fe, #6366f1, #ec4899) !important;
    border-radius: 10px !important;
}

/* Custom styled Table */
table {
    background: rgba(10, 5, 25, 0.45) !important;
    border-radius: 12px !important;
    border-collapse: collapse !important;
    overflow: hidden !important;
    width: 100% !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
}
th {
    background: rgba(139, 92, 246, 0.15) !important;
    color: #e2e8f0 !important;
    padding: 12px 16px !important;
    text-align: left !important;
    font-weight: 600 !important;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1) !important;
}
td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 12px 16px !important;
    color: #cbd5e1 !important;
}

/* ── Share result button ─────────────────────────── */
.share-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: transparent;
  border: 1px solid #1E2D40;
  border-radius: 8px;
  padding: 7px 14px;
  color: #4A6070;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
  font-family: 'DM Sans', sans-serif;
}
.share-btn:hover {
  border-color: #2B7FD4;
  color: #2B7FD4;
  background: rgba(43,127,212,0.08);
}

/* ── Verdict entrance animation ──────────────────── */
@keyframes verdictPop {
  0%   { opacity:0; transform: scale(0.92) translateY(6px); }
  60%  { transform: scale(1.02) translateY(-1px); }
  100% { opacity:1; transform: scale(1) translateY(0); }
}
.result-card {
  animation: verdictPop 0.4s cubic-bezier(0.34,1.56,0.64,1) !important;
}

/* ── Mobile responsive fixes ─────────────────────── */
@media (max-width: 640px) {
  .main .block-container {
    padding: 0.8rem 0.8rem 3rem !important;
  }
  .brand-name {
    font-size: 17px !important;
  }
  .brand-stats {
    display: none !important;
  }
  .stRadio > div > label {
    padding: 6px 8px !important;
    font-size: 11px !important;
  }
  .signal-grid {
    grid-template-columns: repeat(2, 1fr) !important;
  }
  .result-verdict {
    font-size: 20px !important;
  }
}

/* ── Fact-check results card ──────────────────────── */
.factcheck-card {
  background: #111820;
  border: 1px solid #1E2D40;
  border-radius: 8px;
  padding: 12px 14px;
  margin-bottom: 8px;
  transition: border-color 0.15s;
}
.factcheck-card:hover {
  border-color: #2B7FD4;
}

/* ── Smooth page transitions ──────────────────────── */
.stApp {
  transition: background-color 0.3s ease !important;
}
section[data-testid="stSidebar"] {
  transition: width 0.3s ease !important;
}

/* ── Better scrollbar ────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0D1117; }
::-webkit-scrollbar-thumb { background: #1E2D40; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2B7FD4; }
</style>
""", unsafe_allow_html=True)

# ── Custom Colorful Metrics Rendering Functions ──
def render_colorful_cards(signals):
    cards_html = f"""
    <div style="display: flex; gap: 15px; justify-content: space-between; flex-wrap: wrap; margin-top: 15px; margin-bottom: 25px;">
        <div style="flex: 1; min-width: 120px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(6, 182, 212, 0.03) 100%); border-left: 5px solid #06b6d4; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #a5f3fc; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Perplexity</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #06b6d4; margin-top: 5px;">{signals['perplexity']}%</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(236, 72, 153, 0.03) 100%); border-left: 5px solid #ec4899; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #fbcfe8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Burstiness</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #ec4899; margin-top: 5px;">{signals['burstiness']}%</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(249, 115, 22, 0.03) 100%); border-left: 5px solid #f97316; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #ffedd5; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">AI Vocab</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #f97316; margin-top: 5px;">{signals['ai_vocab']}%</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.03) 100%); border-left: 5px solid #a855f7; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #e9d5ff; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Repetition</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #a855f7; margin-top: 5px;">{signals['repetition']}%</div>
        </div>
        <div style="flex: 1; min-width: 120px; background: linear-gradient(135deg, rgba(34, 197, 94, 0.15) 0%, rgba(34, 197, 94, 0.03) 100%); border-left: 5px solid #22c55e; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #dcfce7; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Readability</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #22c55e; margin-top: 5px;">{signals['readability']}%</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)

def render_image_cards(signals):
    cards_html = f"""
    <div style="display: flex; gap: 15px; justify-content: space-between; flex-wrap: wrap; margin-top: 15px; margin-bottom: 25px;">
        <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(6, 182, 212, 0.03) 100%); border-left: 5px solid #06b6d4; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #a5f3fc; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Noise Pattern</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #06b6d4; margin-top: 5px;">{signals['noise_pattern']}%</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(236, 72, 153, 0.03) 100%); border-left: 5px solid #ec4899; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #fbcfe8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">DCT Frequency</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #ec4899; margin-top: 5px;">{signals['dct_frequency']}%</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, rgba(249, 115, 22, 0.15) 0%, rgba(249, 115, 22, 0.03) 100%); border-left: 5px solid #f97316; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #ffedd5; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Colour Smooth.</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #f97316; margin-top: 5px;">{signals['colour_smoothness']}%</div>
        </div>
        <div style="flex: 1; min-width: 140px; background: linear-gradient(135deg, rgba(168, 85, 247, 0.15) 0%, rgba(168, 85, 247, 0.03) 100%); border-left: 5px solid #a855f7; padding: 15px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15); border-top: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.85rem; color: #e9d5ff; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;">Edge Uniformity</div>
            <div style="font-size: 1.8rem; font-weight: 800; color: #a855f7; margin-top: 5px;">{signals['edge_uniformity']}%</div>
        </div>
    </div>
    """
    st.markdown(cards_html, unsafe_allow_html=True)


MODEL_DIR = os.path.join(BASE_DIR, "bert_model")
LOG_FILE = os.path.join(BASE_DIR, "prediction_log.csv")

# ── Cache Sklearn Models (Fast & Lightweight) ────────
@st.cache_resource
def load_sklearn_models():
    lr_path = os.path.join(BASE_DIR, 'lr_model.pkl')
    rf_path = os.path.join(BASE_DIR, 'rf_model.pkl')
    tfidf_path = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    
    models = {}
    try:
        if os.path.exists(lr_path):
            with open(lr_path, 'rb') as f:
                models['LR'] = pickle.load(f)
        if os.path.exists(rf_path):
            with open(rf_path, 'rb') as f:
                models['RF'] = pickle.load(f)
        if os.path.exists(tfidf_path):
            with open(tfidf_path, 'rb') as f:
                models['TFIDF'] = pickle.load(f)
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                models['Scaler'] = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading lightweight models: {e}")
        
    return models

# ── Lazy-Load DistilBERT Models (Deep Learning - Heavy) 
@st.cache_resource
def load_bert_models():
    if not os.path.exists(MODEL_DIR):
        return None, None, None
        
    # Lazy imports to optimize app startup speed
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    
    try:
        device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    except Exception:
        device = torch.device('cpu')
        
    try:
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device)
        model.eval()
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Failed to load DistilBERT: {e}")
        return None, None, None

# Load lightweight models on start (takes <50ms)
sk_models = load_sklearn_models()

# Warm up the model in the background on first load
# so first prediction is instant instead of slow
@st.cache_resource
def warmup_model():
    mdl, tok, device = load_bert_models()
    if mdl is not None and tok is not None:
        import torch
        # Run a dummy prediction to load weights into memory
        inputs = tok("warmup", return_tensors='pt',
                     truncation=True, max_length=10, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = mdl(**inputs)
    return "ready"

warmup_status = warmup_model()  # runs once, cached forever

# ── Logging Setup ───────────────────────────────────
def log_prediction(input_text, label, confidence, model_used):
    file_exists = os.path.exists(LOG_FILE)
    df_len = 0
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['ID', 'Timestamp', 'Input_Text', 'Label', 'Confidence', 'Model_Used', 'User_Feedback'])
        else:
            try:
                df_len = len(pd.read_csv(LOG_FILE))
            except Exception:
                df_len = 0
            
        row_id = df_len + 1
        writer.writerow([
            row_id,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text[:300],  # truncate to save space
            label,
            confidence,
            model_used,
            "" # empty feedback initially
        ])
    return row_id

def update_feedback(row_id, feedback):
    if not os.path.exists(LOG_FILE): return
    try:
        df = pd.read_csv(LOG_FILE)
        if row_id <= len(df):
            df.loc[df['ID'] == row_id, 'User_Feedback'] = feedback
            df.to_csv(LOG_FILE, index=False)
    except Exception:
        pass

# ── Prediction Logic ────────────────────────────────
def predict_sklearn(text, model, tfidf, scaler):
    cleaned = clean_text(text)
    text_vector = tfidf.transform([cleaned])
    
    style_df = extract_features([text])
    style_scaled = scaler.transform(style_df)
    style_sparse = scipy.sparse.csr_matrix(style_scaled)
    
    fused_features = scipy.sparse.hstack([text_vector, style_sparse])
    
    pred = model.predict(fused_features)[0]
    proba = model.predict_proba(fused_features)[0]
    
    label = "REAL" if pred == 1 else "FAKE"
    confidence = round(max(proba) * 100, 1)
    
    if label == "REAL":
        note = "High confidence — Style and semantic patterns indicate a genuine news article."
    else:
        note = "High confidence — Stylistic markers detect signs typical of news manipulation."
        
    return label, confidence, note

def predict_single(text, model_name):
    if len(text.split()) < 6:
        return "UNCERTAIN", 0.0, "Too short — add more text for a reliable result."
        
    if model_name in ('Logistic Regression', 'Random Forest'):
        model_key = 'LR' if model_name == 'Logistic Regression' else 'RF'
        if model_key not in sk_models or 'TFIDF' not in sk_models or 'Scaler' not in sk_models:
            return "MODEL MISSING", 0.0, "Lightweight models not found. Please run the training pipeline first."
        return predict_sklearn(text, sk_models[model_key], sk_models['TFIDF'], sk_models['Scaler'])
        
    elif model_name == 'DistilBERT':
        # Lazy imports for DistilBERT mode
        try:
            import torch
        except ImportError:
            return "IMPORT ERROR", 0.0, "PyTorch is missing from this virtual environment."
            
        model, tokenizer, device = load_bert_models()
        if model is None:
            return "MODEL MISSING", 0.0, "DistilBERT model not found in fake_news_project/bert_model/"

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

def run_batch_prediction_sklearn(texts: list, model, tfidf, scaler) -> pd.DataFrame:
    results = []
    for text in texts:
        label, conf, _ = predict_sklearn(text, model, tfidf, scaler)
        results.append({
            "Verdict": label,
            "Confidence (%)": conf
        })
    df_out = pd.DataFrame({"Analyzed Text": texts})
    df_res = pd.DataFrame(results)
    return pd.concat([df_out, df_res], axis=1)

def _show_result(label, confidence, note, raw_text, model_name):
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
    
    # Check Google Fact Check API for human reviewed checks
    factcheck_results = check_google_factcheck(raw_text[:200])
    render_factcheck_results(factcheck_results)
    
    # Logging and Feedback mechanism
    if "current_log_id" not in st.session_state or st.session_state.get('last_text') != raw_text:
        st.session_state.current_log_id = log_prediction(raw_text, label, confidence, model_name)
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
        
    # ── Share result ───────────────────────────────────
    app_url = "https://web-production-08501d.up.railway.app"  # updated after deploy
    wa_num  = "+14155238886"                                 # Twilio sandbox number
    
    share_text = (
        f"I just fact-checked this with VerifyAI 🔍\n"
        f"Verdict: {label} ({confidence:.0f}% confidence)\n\n"
        f"Check it yourself: {app_url}"
    )
    wa_share_url = f"https://wa.me/?text={share_text.replace(' ','%20').replace('\n','%0A')}"
    
    st.divider()
    col_share, col_copy, _ = st.columns([2, 2, 3])
    
    with col_share:
        st.link_button(
            "📤 Share on WhatsApp",
            wa_share_url,
            use_container_width=True
        )
    with col_copy:
        if st.button("📋 Copy result", use_container_width=True):
            st.write(f"```\n{share_text}\n```")
            st.toast("Copied to clipboard!")

# ── Sidebar Configuration & Controls ─────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    st.markdown("Select which ML classifier to use for general news predictions:")
    
    # Check if models are available
    lr_avail = 'LR' in sk_models
    rf_avail = 'RF' in sk_models
    bert_avail = os.path.exists(MODEL_DIR)
    
    model_options = []
    if lr_avail: model_options.append("Logistic Regression")
    if rf_avail: model_options.append("Random Forest")
    if bert_avail: model_options.append("DistilBERT")
    
    if not model_options:
        st.warning("⚠️ No trained models found! Please run the training pipeline first.")
        model_options = ["None Available"]
        
    selected_model = st.selectbox(
        "Active Classifier",
        model_options,
        index=0 if "Logistic Regression" in model_options else 0
    )
    
    st.divider()
    
    st.markdown("### 📊 Active Model Stats")
    if selected_model == "Logistic Regression":
        st.info("**Type**: Stylistic + TF-IDF Classifier\n\n**Accuracy**: **99.05%**\n\n**Response**: Instant (< 2ms)\n\n**Size**: Lightweight (0.8 MB)")
    elif selected_model == "Random Forest":
        st.info("**Type**: Stylistic + TF-IDF Ensemble\n\n**Accuracy**: **99.00%**\n\n**Response**: Fast (< 15ms)\n\n**Size**: Lightweight (1.8 MB)")
    elif selected_model == "DistilBERT":
        st.info("**Type**: Deep Transformer model\n\n**Accuracy**: **99.61%**\n\n**Response**: Heavy (~50ms)\n\n**Size**: Massive (267.8 MB)")
        
    st.divider()
    
    st.markdown("### 🛠️ System Diagnostics")
    import platform
    st.text(f"OS: {platform.system()} {platform.machine()}")
    if selected_model == "DistilBERT":
        try:
            import torch
            device_str = 'MPS (GPU)' if torch.backends.mps.is_available() else ('CUDA (GPU)' if torch.cuda.is_available() else 'CPU')
            st.text(f"Device: {device_str}")
        except Exception:
            st.text("Device: CPU")
    else:
        st.text("Device: CPU (Optimized)")

# ── Main Header ──────────────────────────────────────
st.title("🔍 Fake News Detector (Pro)")
st.markdown("Powered by Stylistic Feature Fusion and Deep Learning. Check raw articles, Live URLs, or CSV batches!")
st.divider()

# ── 6-Mode Selector ──────────────────────────────────
mode = st.radio(
    "Choose input mode",
    ["📝 Paste text", "🔗 Paste URL", "📂 Upload CSV", "🤖 AI vs Human", "🖼️ AI Image Detector", "📰 Live News Search"],
    horizontal=True
)

if mode == "📝 Paste text":
    user_input = st.text_area("Enter news text here", height=160)
    if st.button("🔎 Analyse", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner(""):
                placeholder = st.empty()
                placeholder.markdown("""
                <div style="background:rgba(255,255,255,0.05);border-radius:12px;
                            padding:20px;animation:pulse 1.5s infinite;">
                  <div style="height:24px;background:#1A2535;border-radius:6px;
                              width:40%;margin-bottom:12px;"></div>
                  <div style="height:14px;background:#1A2535;border-radius:4px;
                              width:80%;margin-bottom:8px;"></div>
                  <div style="height:8px;background:#1A2535;border-radius:4px;
                              width:100%;"></div>
                </div>
                <style>
                @keyframes pulse {
                  0%,100% { opacity:1; } 50% { opacity:0.5; }
                }
                </style>
                """, unsafe_allow_html=True)
                label, conf, note = predict_single(user_input, selected_model)
                placeholder.empty()  # remove skeleton when done
            _show_result(label, conf, note, user_input, selected_model)

elif mode == "🔗 Paste URL":
    url_input = st.text_input("Paste a news article URL:")
    if st.button("🌐 Scrape & Analyse", use_container_width=True):
        if not url_input.strip():
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Fetching and cleaning article content..."):
                article = scrape_article(url_input)
            
            if article['error']:
                st.error(article['error'])
            else:
                st.info(f"**Found Article:** {article['title']}")
                with st.spinner("Performing deep semantic analysis..."):
                    label, conf, note = predict_single(article['text'], selected_model)
                _show_result(label, conf, note, article['text'], selected_model)

elif mode == "📂 Upload CSV":
    st.info("Upload a CSV file containing headlines to analyze them all at once. The column should be named 'text' or 'headline'.")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file and st.button("🚀 Run Batch Predictor", use_container_width=True):
        with st.spinner("Processing batch predictor..."):
            try:
                df_upload = pd.read_csv(uploaded_file)
                col_name = 'text' if 'text' in df_upload.columns else ('headline' if 'headline' in df_upload.columns else df_upload.columns[0])
                
                texts = df_upload[col_name].dropna().astype(str).tolist()
                
                # Predict according to selected model
                if selected_model in ('Logistic Regression', 'Random Forest'):
                    model_key = 'LR' if selected_model == 'Logistic Regression' else 'RF'
                    df_results = run_batch_prediction_sklearn(
                        texts, sk_models[model_key], sk_models['TFIDF'], sk_models['Scaler']
                    )
                else:
                    import torch
                    model, tokenizer, device = load_bert_models()
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
            except Exception as e:
                st.error(f"Error processing CSV file: {e}")

elif mode == "🤖 AI vs Human":
    st.info("Check if a text was generated by AI models like ChatGPT, Claude, or Gemini.")
    user_text = st.text_area("Enter text to analyze here", height=160)
    
    if st.button("🧠 Analyze AI Patterns", use_container_width=True):
        if not user_text.strip():
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Analyzing linguistic entropy and signals..."):
                res = detect_ai_text(user_text)
            
            # Show verdict
            st.divider()
            if res['verdict'] == "HUMAN WRITTEN": st.success(f"### ✅ {res['verdict']}")
            elif res['verdict'] == "AI GENERATED": st.error(f"### 🤖 {res['verdict']}")
            elif res['verdict'] in ["LIKELY HUMAN", "LIKELY AI", "UNCERTAIN"]: st.warning(f"### ⚠️ {res['verdict']}")
            
            st.markdown(f"**{res['note']}**")
            st.markdown(f"**AI Score: {res['ai_score']}%** (Words: {res['word_count']})")
            st.progress(max(1, min(int(res['ai_score']), 100)))
            
            st.markdown("### 📊 Signal Breakdown")
            render_colorful_cards(res['signals'])
            
            st.markdown("### 🔍 Flagged Words")
            if res['flagged_words']:
                badges = " ".join([f"`{w}`" for w in res['flagged_words']])
                st.markdown(badges)
            else:
                st.markdown("*No common AI words detected.*")
                
            # Rewrite suggestions
            if res['flagged_words']:
                with st.expander("🛠️ How to make it more human-like"):
                    st.markdown("Try swapping the flagged words for these more natural alternatives:")
                    tips_data = get_rewrite_tips(user_text, res['ai_score'], res['flagged_words'])
                    
                    df_tips = pd.DataFrame([
                        {"Word": w, "Alternatives": ", ".join(alts)} 
                        for w, alts in tips_data['tips'].items()
                    ])
                    st.table(df_tips)
                    
                    st.success(f"**Expected new AI Score:** ~{tips_data['estimated_new_score']}%")

elif mode == "🖼️ AI Image Detector":
    st.info("Upload an image to check if it was generated by AI or captured with a real camera.")
    uploaded_img = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_img and st.button("🔍 Analyze Image", use_container_width=True):
        with st.spinner("Extracting pixel-level frequency signals..."):
            img_bytes = uploaded_img.read()
            res = detect_ai_image(img_bytes)
            
            st.image(img_bytes, caption=f"{res['image_size'][0]}x{res['image_size'][1]}  •  {res['megapixels']} MP", use_container_width=True)
            
            st.divider()
            if res['verdict'] == "REAL PHOTO": st.success(f"### 📷 {res['verdict']}")
            elif res['verdict'] == "AI GENERATED": st.error(f"### 🤖 {res['verdict']}")
            elif res['verdict'] in ["LIKELY REAL", "LIKELY AI", "UNCERTAIN"]: st.warning(f"### ⚠️ {res['verdict']}")
            
            st.markdown(f"**{res['note']}**")
            st.markdown(f"**AI Score: {res['ai_score']}%**")
            st.progress(max(1, min(int(res['ai_score']), 100)))
            
            st.markdown("### 📊 Signal Breakdown")
            render_image_cards(res['signals'])
            
            st.markdown("### 🏷️ Metadata")
            if res['metadata']['has_exif']:
                st.code(f"Camera: {res['metadata']['camera']}\nSoftware: {res['metadata']['software']}")
            else:
                st.markdown("*No EXIF camera metadata found (Very common in AI).*")

elif mode == "📰 Live News Search":
    st.info("Search a topic to fetch live Google News headlines and evaluate their authenticity.")
    query = st.text_input("Enter search topic:")
    
    if st.button("🔍 Search News", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a topic.")
        else:
            with st.spinner("Fetching live Google News RSS feed..."):
                results = search_google_news(query, num_results=5)
                st.session_state.live_search_results = results
                
    if 'live_search_results' in st.session_state and st.session_state.live_search_results:
        results = st.session_state.live_search_results
        st.write(f"Found {len(results)} recent articles:")
        
        for idx, result in enumerate(results):
            st.markdown(f"**{result['title']}**")
            st.caption(f"Published: {result['published']}")
            st.caption(f"[Link to Source]({result['link']})")
            
            if st.button(f"Analyse Article #{idx+1}", key=f"analyse_{idx}"):
                with st.spinner("Fetching and evaluating..."):
                    article_data = scrape_article(result['link'])
                    if not article_data['error']:
                        label, conf, note = predict_single(article_data['text'], selected_model)
                        _show_result(label, conf, note, article_data['text'], selected_model)
                    else:
                        st.warning("Failed to extract full article text. Evaluating headline instead.")
                        label, conf, note = predict_single(result['title'], selected_model)
                        _show_result(label, conf, note, result['title'], selected_model)
            st.divider()

# ── Footer ───────────────────────────────────────────
st.divider()
st.caption(f"Active Model: {selected_model} · Accuracy up to 99.61%")
