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
    if not body or body.lower() in help_commands:
        return formatter.fmt_help(lang)
        
    # Pre-classify message type
    classification = classify_message(body)
    msg_type = classification['type']
    
    if msg_type == 'too_short':
        verdicts = {
            'english': 'UNCERTAIN',
            'hindi': 'अनिश्चित',
            'telugu': 'అనిశ్చితం'
        }
        notes = {
            'english': 'Text is too short (under 6 words). Please provide more context or a complete sentence for a reliable stylistic check.',
            'hindi': 'पाठ बहुत छोटा है (6 शब्दों से कम)। कृपया विश्वसनीय शैली जांच के लिए अधिक संदर्भ या पूरा वाक्य प्रदान करें।',
            'telugu': 'పాఠం చాలా చిన్నదిగా ఉంది (6 పదాల కంటే తక్కువ)। దయచేసి నమ్మదగిన శైలి తనిఖీ కోసం మరింత సమాచారం లేదా పూర్తి వాక్యాన్ని అందించండి।'
        }
        return formatter.format_fake_news(body, verdicts.get(lang, verdicts['english']), 0.0, notes.get(lang, notes['english']), lang=lang)
        
    elif msg_type == 'question_about_ad':
        verdicts = {
            'english': 'AD / PROMO QUERY',
            'hindi': 'विज्ञापन प्रश्न',
            'telugu': 'ప్రకటన ప్రశ్న'
        }
        notes = {
            'english': 'This query appears to be about an advertisement or brand promotion. VerifyAI evaluates news claims rather than commercial advertisements.',
            'hindi': 'यह प्रश्न किसी विज्ञापन या ब्रांड प्रचार के बारे में प्रतीत होता है। VerifyAI व्यावसायिक विज्ञापनों के बजाय समाचार दावों का मूल्यांकन करता है।',
            'telugu': 'ఈ ప్రశ్న ప్రకటన లేదా బ్రాండ్ ప్రమోషన్ గురించి ఉన్నట్లు అనిపిస్తుంది। VerifyAI కేవలం వార్తలను మాత్రమే విశ్లేషిస్తుంది, వాణిజ్య ప్రకటనలను కాదు।'
        }
        return formatter.format_fake_news(body, verdicts.get(lang, verdicts['english']), 0.0, notes.get(lang, notes['english']), lang=lang)
        
    elif msg_type == 'question':
        verdicts = {
            'english': 'GENERAL QUESTION',
            'hindi': 'सामान्य प्रश्न',
            'telugu': 'సాధారణ ప్రశ్న'
        }
        notes = {
            'english': 'This looks like a general question. VerifyAI is optimized to verify specific rumors and news claims. If this is a rumor, please rephrase it as a factual statement.',
            'hindi': 'यह एक सामान्य प्रश्न लगता है। VerifyAI अफवाहों और समाचार दावों की पुष्टि के लिए अनुकूलित है। यदि यह एक अफवाह है, तो कृपया इसे एक कथन के रूप में फिर से लिखें।',
            'telugu': 'ఇది ఒక సాధారణ ప్రశ్నలా అనిపిస్తుంది। VerifyAI నిర్దిష్ట పుకార్లు మరియు వార్తా దావాలను ధృవీకరించడానికి మాత్రమే సహాయపడుతుంది। ఇది వార్త అయితే దయచేసి దానిని ఒక ప్రకటనగా తిరిగి రాయండి।'
        }
        return formatter.format_fake_news(body, verdicts.get(lang, verdicts['english']), 0.0, notes.get(lang, notes['english']), lang=lang)
        
    elif msg_type == 'advertisement':
        verdicts = {
            'english': 'ADVERTISEMENT',
            'hindi': 'विज्ञापन',
            'telugu': 'ప్రకటన'
        }
        notes = {
            'english': 'This content has been identified as promotional or marketing material. Our verification engine only evaluates news articles and social media rumors.',
            'hindi': 'यह सामग्री विज्ञापन या प्रचार प्रतीत होती है। हमारा सत्यापन इंजन केवल समाचार लेखों और सोशल मीडिया अफवाहों का मूल्यांकन करता है।',
            'telugu': 'ఈ కంటెంట్ ఒక ప్రకటన లేదా మార్కెటింగ్ సమాచారంగా గుర్తించబడింది। మా ధృవీకరణ వ్యవస్థ కేవలం వార్తలు మరియు పుకార్లను మాత్రమే విశ్లేషిస్తుంది।'
        }
        return formatter.format_fake_news(body, verdicts.get(lang, verdicts['english']), 0.0, notes.get(lang, notes['english']), lang=lang)
        
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
