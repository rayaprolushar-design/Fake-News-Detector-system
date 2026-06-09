import os
import sys
import pickle
import scipy.sparse
import requests
import re
import io
from urllib.parse import urlparse
from PIL import Image

# Add project folder to sys.path to allow internal imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "fake_news_project")
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from text_processing import clean_text
from features import extract_features
from url_scraper import scrape_article
from ai_detector import detect_ai_text
from image_detector import detect_ai_image
from lang_detector import detect_language
from multilingual_predictor import predict_multilingual
from message_classifier import classify_message

# Twilio Sandbox credentials (configured via .env file)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "ACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "your_auth_token_here")

# Predefined source credibility dictionary for Indian and global news domains
SOURCE_CREDIBILITY = {
    # Indian News - High Credibility
    "ndtv.com": {"rating": "TRUSTED", "score": 88, "type": "Mainstream News"},
    "thehindu.com": {"rating": "TRUSTED", "score": 92, "type": "Mainstream News"},
    "timesofindia.indiatimes.com": {"rating": "TRUSTED", "score": 85, "type": "Mainstream News"},
    "indianexpress.com": {"rating": "TRUSTED", "score": 90, "type": "Mainstream News"},
    "hindustantimes.com": {"rating": "TRUSTED", "score": 87, "type": "Mainstream News"},
    "pib.gov.in": {"rating": "TRUSTED", "score": 98, "type": "Government Portal"},
    "altnews.in": {"rating": "TRUSTED", "score": 95, "type": "Fact Checker"},
    "boomlive.in": {"rating": "TRUSTED", "score": 95, "type": "Fact Checker"},
    "ptinews.com": {"rating": "TRUSTED", "score": 94, "type": "News Agency"},
    
    # Global News - High Credibility
    "bbc.com": {"rating": "TRUSTED", "score": 94, "type": "Public Broadcaster"},
    "bbc.co.uk": {"rating": "TRUSTED", "score": 94, "type": "Public Broadcaster"},
    "reuters.com": {"rating": "TRUSTED", "score": 97, "type": "News Agency"},
    "apnews.com": {"rating": "TRUSTED", "score": 96, "type": "News Agency"},
    "nytimes.com": {"rating": "TRUSTED", "score": 92, "type": "Mainstream News"},
    "washingtonpost.com": {"rating": "TRUSTED", "score": 91, "type": "Mainstream News"},
    "theguardian.com": {"rating": "TRUSTED", "score": 90, "type": "Mainstream News"},
    "guardian.co.uk": {"rating": "TRUSTED", "score": 90, "type": "Mainstream News"},
    "bloomberg.com": {"rating": "TRUSTED", "score": 93, "type": "Financial News"},
    "npr.org": {"rating": "TRUSTED", "score": 92, "type": "Public Broadcaster"},
    
    # Questionable / Tabloid / Heavy Bias
    "dailymail.co.uk": {"rating": "QUESTIONABLE", "score": 45, "type": "Tabloid / High Bias"},
    "thesun.co.uk": {"rating": "QUESTIONABLE", "score": 35, "type": "Tabloid"},
    "breitbart.com": {"rating": "QUESTIONABLE", "score": 30, "type": "Extreme Bias / Misinformation"},
    "infowars.com": {"rating": "QUESTIONABLE", "score": 10, "type": "Conspiracy / Fake News"},
    "rt.com": {"rating": "QUESTIONABLE", "score": 40, "type": "State-Controlled News"},
    "sputniknews.com": {"rating": "QUESTIONABLE", "score": 38, "type": "State-Controlled News"},
    "opindia.com": {"rating": "QUESTIONABLE", "score": 45, "type": "Heavy Bias"},
    "postcard.news": {"rating": "QUESTIONABLE", "score": 15, "type": "Fake News Site"},
}

def load_sklearn_models():
    lr_path = os.path.join(PROJECT_DIR, 'lr_model.pkl')
    rf_path = os.path.join(PROJECT_DIR, 'rf_model.pkl')
    tfidf_path = os.path.join(PROJECT_DIR, 'tfidf_vectorizer.pkl')
    scaler_path = os.path.join(PROJECT_DIR, 'scaler.pkl')
    
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
        print(f"Error loading lightweight models: {e}")
        
    return models

# Load lightweight models globally for performance
SK_MODELS = load_sklearn_models()

# Global variables to cache DistilBERT
BERT_MODEL = None
BERT_TOKENIZER = None
BERT_DEVICE = None

def get_bert_model():
    """Lazily load the DistilBERT model to save startup memory if it is not requested."""
    global BERT_MODEL, BERT_TOKENIZER, BERT_DEVICE
    
    model_dir = os.path.join(PROJECT_DIR, "bert_model")
    if not os.path.exists(model_dir):
        return None, None, None
        
    if BERT_MODEL is None:
        try:
            import torch
            from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
            
            try:
                device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
            except Exception:
                device = torch.device('cpu')
                
            model = DistilBertForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
            model.to(device)
            model.eval()
            tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir, local_files_only=True)
            
            BERT_MODEL = model
            BERT_TOKENIZER = tokenizer
            BERT_DEVICE = device
        except Exception as e:
            print(f"Failed to load DistilBERT: {e}")
            
    return BERT_MODEL, BERT_TOKENIZER, BERT_DEVICE

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
        note = "Style and semantic patterns indicate a genuine news article."
    else:
        note = "Stylistic markers detect patterns typical of news manipulation."
        
    return label, confidence, note

def predict_sklearn_fallback(text, extra_note=None):
    if 'LR' in SK_MODELS and 'TFIDF' in SK_MODELS and 'Scaler' in SK_MODELS:
        label, conf, note = predict_sklearn(text, SK_MODELS['LR'], SK_MODELS['TFIDF'], SK_MODELS['Scaler'])
        if extra_note:
            note = f"{note}\n\n*System Note:* {extra_note}"
        return label, conf, note
    else:
        return "MODEL MISSING", 0.0, "Model files not found. Please train models first."

def predict_bert(text):
    """Predict text credibility using DistilBERT model. Fallback to Logistic Regression if missing."""
    model, tokenizer, device = get_bert_model()
    if model is None:
        return predict_sklearn_fallback(text, "DistilBERT model not loaded. Evaluated with Logistic Regression model.")
        
    try:
        import torch
        inputs = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            
        fake_prob = probs[0].item() * 100
        real_prob = probs[1].item() * 100
        
        if real_prob >= 75:
            label, note = "REAL", "DistilBERT model identifies this article content as genuine."
        elif fake_prob >= 75:
            label, note = "FAKE", "DistilBERT model detects strong markers of misinformation."
        elif real_prob >= 60:
            label, note = "LIKELY REAL", "Leaning real, but verify with a trusted source."
        elif fake_prob >= 60:
            label, note = "LIKELY FAKE", "Leaning fake, but check before sharing."
        else:
            label, note = "UNCERTAIN", "The deep-learning model isn't confident — verify manually."
            
        return label, round(max(real_prob, fake_prob), 1), note
    except Exception as e:
        return predict_sklearn_fallback(text, f"DistilBERT evaluation failed: {e}. Fallback to Logistic Regression used.")

def download_twilio_media(media_url: str) -> bytes:
    """Download media from Twilio. Handles HTTP Basic Authentication if credentials are set."""
    auth = None
    if TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and not TWILIO_ACCOUNT_SID.startswith("ACXXXX"):
        auth = (TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
    resp = requests.get(media_url, auth=auth, timeout=20)
    resp.raise_for_status()
    return resp.content

def handle_message(body: str, media_url: str = None, media_type: str = None) -> str:
    """
    Main routing function for the WhatsApp bot.
    Dispatches:
      1. Image -> AI Image Detector
      2. URL -> Scraper + Domain Credibility + BERT
      3. /ai text -> AI Text Detector
      4. General text -> Fake News Classifier (Logistic Regression)
    """
    body = (body or "").strip()
    import formatter
    
    # ── 1. Image Detector Dispatcher ────────────────────────────────
    if media_url and media_type and media_type.startswith("image/"):
        try:
            img_bytes = download_twilio_media(media_url)
            analysis = detect_ai_image(img_bytes)
            return formatter.format_ai_image(analysis)
        except Exception as e:
            return f"❌ *Error analyzing image:* {str(e)}"
            
    # ── 2. URL Scraper & BERT Dispatcher ──────────────────────────
    urls = re.findall(r'(https?://[^\s]+)', body)
    if urls:
        url = urls[0]
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
                
            # Check domain credibility
            credibility = {"rating": "UNKNOWN", "score": "N/A", "type": "Unverified Source"}
            matched_domain = domain
            for k, v in SOURCE_CREDIBILITY.items():
                if domain == k or domain.endswith("." + k):
                    credibility = v
                    matched_domain = k
                    break
                    
            # Scrape article
            scrape_res = scrape_article(url)
            if scrape_res.get('error'):
                # Domain info available, content scrape failed
                err_msg = scrape_res['error']
                return (
                    f"⚠️ *Article Extraction Failed*\n"
                    f"{formatter.format_divider()}\n"
                    f"*URL:* {url}\n\n"
                    f"*Source Credibility:*\n"
                    f"• *Domain:* `{matched_domain}`\n"
                    f"• *Trust Level:* {'🟢 *TRUSTED*' if credibility['rating'] == 'TRUSTED' else '🔴 *QUESTIONABLE*' if credibility['rating'] == 'QUESTIONABLE' else '🟡 *UNKNOWN*'}\n"
                    f"• *Score:* *{credibility['score']}/100*\n\n"
                    f"*Scraping Info:* {err_msg}\n"
                    f"_(Fallback to headline analysis not possible due to network or site blockers)_"
                    f"{formatter.format_footer()}"
                )
                
            article_title = scrape_res.get('title', 'Unknown Title')
            article_text = scrape_res.get('text', '')
            
            # Classify content with DistilBERT
            label, confidence, note = predict_bert(article_text)
            
            return formatter.format_url_analysis(
                url, article_title, matched_domain, credibility, label, confidence, note
            )
        except Exception as e:
            return f"❌ *Error processing URL:* {str(e)}"
            
    # ── 3. AI Text Detector Command Dispatcher ──────────────────────
    if body.lower().startswith("/ai"):
        ai_text_query = body[3:].strip()
        if not ai_text_query:
            return (
                f"🤖 *AI Text Detector*\n"
                f"{formatter.format_divider()}\n"
                f"Usage: Send `/ai <text to analyze>` to check if the text is AI-written."
                f"{formatter.format_footer()}"
            )
        try:
            analysis = detect_ai_text(ai_text_query)
            return formatter.format_ai_text(ai_text_query, analysis)
        except Exception as e:
            return f"❌ *Error analyzing AI text:* {str(e)}"
            
    # ── 4. General Fake News Detector Fallback ─────────────────────
    lang = detect_language(body)
    
    # Check for help/welcome commands
    help_commands = {
        'help', 'info', 'menu', 'guide', 'welcome', 
        'मदद', 'सहायता', 'जानकारी', 'मेन्यू',
        'సహాయం', 'సమాచారం', 'మెనూ'
    }
    
    if len(body) < 4:
        if body.lower() in help_commands:
            return formatter.fmt_help(lang)
        quick_helps = {
            'english': "Send me a news headline or message to check.\n\nType *help* to see all commands.",
            'hindi': "जांचने के लिए मुझे एक समाचार शीर्षक या संदेश भेजें।\n\nसभी कमांड देखने के लिए *मदद* टाइप करें।",
            'telugu': "తనిఖీ చేయడానికి నాకు వార్తా శీర్షిక లేదా సందేశాన్ని పంపండి।\n\nఅన్ని ఆదేశాలను చూడటానికి *సహాయం* టైప్ చేయండి।"
        }
        return quick_helps.get(lang, quick_helps['english'])

    if body.lower() in help_commands:
        return formatter.fmt_help(lang)
        
    # Pre-classify message type
    classification = classify_message(body)
    msg_type = classification['type']
    
    if msg_type == 'too_short':
        texts = {
            'english': "ℹ️ *Too short to analyse.*\n\nPlease send a complete news headline or message — at least 6–8 words.",
            'hindi': "ℹ️ *विश्लेषण के लिए बहुत छोटा है।*\n\nकृपया एक पूरा समाचार शीर्षक या संदेश भेजें — कम से कम 6-8 शब्द।",
            'telugu': "ℹ️ *విశ్లేషించడానికి చాలా చిన్నదిగా ఉంది।*\n\nదయచేసి పూర్తి వార్తా శీర్షిక లేదా సందేశాన్ని పంపండి — కనీసం 6-8 పదాలు."
        }
        return texts.get(lang, texts['english'])
        
    elif msg_type == 'question_about_ad':
        texts = {
            'english': "ℹ️ *This looks like a question about an advertisement.*\n\nI'm designed to check *news claims* for misinformation — not evaluate ads or branded content.\n\nIf you want to check whether a news story is fake, send me the headline or article text.\n\nType *help* to see what I can check.",
            'hindi': "ℹ️ *यह एक विज्ञापन के बारे में प्रश्न प्रतीत होता है।*\n\nमुझे गलत सूचनाओं के लिए *समाचार दावों* की जांच करने के लिए डिज़ाइन किया गया है — विज्ञापनों या ब्रांडेड सामग्री का मूल्यांकन करने के लिए नहीं।\n\nयदि आप जांचना चाहते हैं कि कोई *समाचार* फर्जी है या नहीं, तो मुझे शीर्षक या लेख का पाठ भेजें।\n\nक्या जांच कर सकता हूं, यह देखने के लिए *मदद* टाइप करें।",
            'telugu': "ℹ️ *ఇది ఒక ప్రకటన గురించిన ప్రశ్నలా అనిపిస్తుంది।*\n\nనేను తప్పుడు సమాచారం కోసం *వార్తా దావాలను* తనిఖీ చేయడానికి రూపొందించబడ్డాను – ప్రకటనలు లేదా బ్రాండెడ్ కంటెంట్‌ను విశ్లేషించడానికి కాదు।\n\nవార్త నిజమో కాదో తనిఖీ చేయాలనుకుంటే, నాకు శీర్షిక లేదా వ్యాస పాఠాన్ని పంపండి।\n\nనేను ఏమి తనిఖీ చేయగలనో చూడటానికి *సహాయం* టైప్ చేయండి।"
        }
        return texts.get(lang, texts['english'])
        
    elif msg_type == 'question':
        texts = {
            'english': "ℹ️ *I can't answer general questions.*\n\nI'm built specifically to check whether a *news claim or viral message* is fake or real.\n\nSend me an actual news headline or a forwarded WhatsApp message and I'll analyse it.\n\nExample:\n_SHOCKING!! Vaccine causes side effects, doctors hiding truth!!_",
            'hindi': "ℹ️ *मैं सामान्य प्रश्नों के उत्तर नहीं दे सकता।*\n\nमैं विशेष रूप से यह जांचने के लिए बनाया गया हूं कि कोई *समाचार दावा या वायरल संदेश* नकली है या असली।\n\nमुझे एक वास्तविक समाचार शीर्षक या फ़ॉरवर्ड किया गया व्हाट्सएप संदेश भेजें और मैं उसका विश्लेषण करूंगा।\n\nउदाहरण:\n_चौंकाने वाला!! वैक्सीन से साइड इफेक्ट होते हैं, डॉक्टर छुपा रहे हैं सच!!_",
            'telugu': "ℹ️ *నేను సాధారణ ప్రశ్నలకు సమాధానం చెప్పలేను।*\n\nవార్తా దావా లేదా వైరల్ సందేశం నిజమో కాదో తనిఖీ చేయడానికి నేను ప్రత్యేకంగా రూపొందించబడ్డాను।\n\nనాకు అసలు వార్తా శీర్షిక లేదా ఫార్వార్డ్ చేసిన వాట్సాప్ సందేశాన్ని పంపండి మరియు నేను దానిని విశ్లేషిస్తాను।\n\nఉదాహరణ:\n_షాకింగ్!! వ్యాక్సిన్ వల్ల దుష్ప్రభావాలు కలుగుతాయి, వైద్యులు నిజాన్ని దాస్తున్నారు!!_"
        }
        return texts.get(lang, texts['english'])
        
    elif msg_type == 'advertisement':
        texts = {
            'english': "ℹ️ *This looks like advertisement content.*\n\nI check news and viral claims for misinformation — not commercial ads.\n\nIf you received a suspicious *news message* on WhatsApp, send that instead and I'll check it.",
            'hindi': "ℹ️ *यह विज्ञापन सामग्री प्रतीत होती है।*\n\nमैं गलत सूचना के लिए समाचार और वायरल दावों की जांच करता हूं — व्यावसायिक विज्ञापनों की नहीं।\n\nयदि आपको व्हाट्सएप पर कोई संदिग्ध *समाचार संदेश* मिला है, तो इसके बजाय उसे भेजें और मैं उसकी जांच करूंगा।",
            'telugu': "ℹ️ *ఇది ప్రకటన కంటెంట్‌లా అనిపిస్తుంది।*\n\nనేను తప్పుడు సమాచారం కోసం వార్తలు మరియు వైరల్ దావాలను తనిఖీ చేస్తాను — వాణిజ్య ప్రకటనలను కాదు।\n\nమీరు వాట్సాప్‌లో ఏదైనా అనుమానాస్పద *వార్తా సందేశాన్ని* అందుకుంటే, దానికి బదులుగా దానిని పంపండి మరియు నేను దానిని తనిఖీ చేస్తాను।"
        }
        return texts.get(lang, texts['english'])
        
    try:
        res = predict_multilingual(body)
        return formatter.format_fake_news(
            body, 
            res['label_native'], 
            res['confidence'], 
            res['note'], 
            lang=res['language']
        )
    except Exception as e:
        return f"❌ *Error analyzing text:* {str(e)}"
