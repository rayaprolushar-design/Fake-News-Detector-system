# multilingual_predictor.py
# Smart router — English → DistilBERT, Hindi/Telugu → IndicBERT (with translation fallback)

import os
import sys
import torch
from transformers import (
    DistilBertTokenizerFast, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
from lang_detector import detect_language
from deep_translator import GoogleTranslator

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_DIR = os.path.join(BASE_DIR, "bert_model")
INDIC_DIR = os.path.join(BASE_DIR, "indic_model")

# Lazy load models
_english_tok   = None
_english_model = None
_indic_tok     = None
_indic_model   = None
_indic_available = None

def _get_english_model():
    global _english_tok, _english_model
    if _english_tok is None:
        try:
            _english_tok   = DistilBertTokenizerFast.from_pretrained(BERT_DIR, local_files_only=True)
            _english_model = DistilBertForSequenceClassification.from_pretrained(BERT_DIR, local_files_only=True)
            _english_model.eval()
        except Exception as e:
            print(f"Error loading English DistilBERT: {e}")
    return _english_tok, _english_model

def _get_indic_model():
    global _indic_tok, _indic_model, _indic_available
    if _indic_available is None:
        if os.path.exists(INDIC_DIR):
            try:
                _indic_tok   = AutoTokenizer.from_pretrained(INDIC_DIR, local_files_only=True)
                _indic_model = AutoModelForSequenceClassification.from_pretrained(INDIC_DIR, local_files_only=True)
                _indic_model.eval()
                _indic_available = True
                print("IndicBERT loaded successfully!")
            except Exception as e:
                print(f"Error loading IndicBERT: {e}")
                _indic_available = False
        else:
            _indic_available = False
    return _indic_tok, _indic_model, _indic_available

# Response templates in 3 languages
VERDICTS = {
    'english': {
        'REAL': ("REAL",        "High confidence — this looks real."),
        'FAKE': ("FAKE",        "Signs of misinformation detected."),
        'LIKELY REAL': ("LIKELY REAL", "Leaning real — verify anyway."),
        'LIKELY FAKE': ("LIKELY FAKE", "Leaning fake — check before sharing."),
        'UNCERTAIN':   ("UNCERTAIN",   "Mixed signals — verify manually."),
    },
    'hindi': {
        'REAL': ("सच्ची खबर",     "यह खबर सच्ची लगती है।"),
        'FAKE': ("फेक न्यूज़",    "यह खबर झूठी हो सकती है। शेयर करने से पहले जांचें।"),
        'LIKELY REAL': ("शायद सच",     "यह सच लगती है — फिर भी जांचें।"),
        'LIKELY FAKE': ("शायद झूठ",    "यह झूठी लग रही है — शेयर न करें।"),
        'UNCERTAIN':   ("अनिश्चित",   "पक्का नहीं — किसी भरोसेमंद स्रोत से जांचें।"),
    },
    'telugu': {
        'REAL': ("నిజమైన వార్త", "ఇది నిజమైన వార్త అని అనిపిస్తోంది।"),
        'FAKE': ("నకిలీ వార్త",  "ఇది తప్పుడు వార్త కావచ్చు। షేర్ చేయడానికి ముందు తనిఖీ చేయండి।"),
        'LIKELY REAL': ("బహుశా నిజం",  "నిజమైనదిగా కనిపిస్తోంది — అయినా తనిఖీ చేయండి।"),
        'LIKELY FAKE': ("బహుశా తప్పు",  "తప్పుగా కనిపిస్తోంది — షేర్ చేయవద్దు।"),
        'UNCERTAIN':   ("అనిశ్చితం",  "నిర్ధారించడం కష్టంగా ఉంది — తనిఖీ చేయండి।"),
    }
}

def predict_english_text(text: str) -> dict:
    """Classifies English text using DistilBERT or fallback Sklearn classifiers."""
    tok, model = _get_english_model()
    if model is None:
        # Fallback to Sklearn Logistic Regression in bot.py
        try:
            from bot import predict_sklearn_fallback
            label, conf, note = predict_sklearn_fallback(text, "English DistilBERT unavailable.")
            return {
                'label': label,
                'label_native': label,
                'confidence': conf,
                'note': note
            }
        except Exception as se:
            return {
                'label': 'UNCERTAIN',
                'label_native': 'UNCERTAIN',
                'confidence': 0.0,
                'note': f"Model fallback failed: {se}"
            }
            
    try:
        inputs = tok(
            text, return_tensors='pt',
            truncation=True, max_length=200, padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
        
        real_p = probs[1].item() * 100
        fake_p = probs[0].item() * 100
        
        if real_p >= 75: key = 'REAL'
        elif fake_p >= 75: key = 'FAKE'
        elif real_p >= 60: key = 'LIKELY REAL'
        elif fake_p >= 60: key = 'LIKELY FAKE'
        else:              key = 'UNCERTAIN'
        
        label_native, note = VERDICTS['english'][key]
        return {
            'label': key,
            'label_native': label_native,
            'confidence': round(max(real_p, fake_p), 1),
            'note': note
        }
    except Exception as e:
        return {
            'label': 'UNCERTAIN',
            'label_native': 'UNCERTAIN',
            'confidence': 0.0,
            'note': f"Prediction error: {e}"
        }

def predict_multilingual(text: str) -> dict:
    """
    Auto-detect language → route to right model → return verdict in same language.
    If IndicBERT is not yet available, automatically translates queries to English,
    runs the classification, and translates the verdict details back.
    """
    lang = detect_language(text)
    
    # Map non-supported languages to English
    if lang not in ('hindi', 'telugu', 'english'):
        lang = 'english'
        
    # ── 1. Hindi/Telugu Model Routing ──────────────────
    if lang in ('hindi', 'telugu'):
        tok, model, is_available = _get_indic_model()
        if is_available:
            try:
                inputs = tok(
                    text, return_tensors='pt',
                    truncation=True, max_length=200, padding=True
                )
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                
                real_p = probs[1].item() * 100
                fake_p = probs[0].item() * 100
                
                if real_p >= 75: key = 'REAL'
                elif fake_p >= 75: key = 'FAKE'
                elif real_p >= 60: key = 'LIKELY REAL'
                elif fake_p >= 60: key = 'LIKELY FAKE'
                else:              key = 'UNCERTAIN'
                
                label_native, note = VERDICTS[lang][key]
                return {
                    'label'        : key,
                    'label_native' : label_native,
                    'confidence'   : round(max(real_p, fake_p), 1),
                    'note'         : note,
                    'language'     : lang,
                    'fallback_used': False
                }
            except Exception as e:
                print(f"IndicBERT classification error: {e}. Falling back to translation.")
                
        # ── 2. Automatic Translation Fallback ──────────────
        try:
            # Map target lang for deep-translator
            src_code = 'hi' if lang == 'hindi' else 'te'
            translated_text = GoogleTranslator(source=src_code, target='en').translate(text)
        except Exception as e:
            # Best effort: use untranslated text if translation fails
            translated_text = text
            
        # Classify translated text with English model
        res_eng = predict_english_text(translated_text)
        key = res_eng['label']
        confidence = res_eng['confidence']
        
        # Get localized verdict labels
        label_native, default_note = VERDICTS[lang][key]
        
        # Translate the detailed analysis note back to the native language
        try:
            tgt_code = 'hi' if lang == 'hindi' else 'te'
            translated_note = GoogleTranslator(source='en', target=tgt_code).translate(res_eng['note'])
        except Exception:
            translated_note = default_note
            
        return {
            'label'        : key,
            'label_native' : label_native,
            'confidence'   : confidence,
            'note'         : translated_note,
            'language'     : lang,
            'fallback_used': True
        }
        
    # ── 3. English Routing (Standard) ──────────────────
    else:
        res_eng = predict_english_text(text)
        return {
            'label'        : res_eng['label'],
            'label_native' : res_eng['label_native'],
            'confidence'   : res_eng['confidence'],
            'note'         : res_eng['note'],
            'language'     : 'english',
            'fallback_used': False
        }

if __name__ == '__main__':
    tests = [
        "Scientists confirm vaccine is 95% effective in trials",
        "यह खबर बिल्कुल झूठी है और लोगों को गुमराह करती है",
        "ఈ వ్యాక్సిన్ వల్ల 5G కనెక్షన్ వస్తుంది అని నిపుణులు చెప్పారు",
    ]
    print("Testing Smart Multilingual Predictor:")
    for t in tests:
        r = predict_multilingual(t)
        fallback_str = " (Fallback translation active)" if r['fallback_used'] else ""
        print(f"[{r['language'].upper()}] Verdict: {r['label_native']} ({r['confidence']}%) {fallback_str}")
        print(f"       Detail: {r['note']}\n")
