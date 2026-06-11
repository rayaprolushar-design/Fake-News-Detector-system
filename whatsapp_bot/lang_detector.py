# lang_detector.py
# Detects whether text is English, Hindi, Telugu, or other

import re
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42   # makes detection deterministic

# Unicode ranges for Indian scripts
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')   # Hindi
TELUGU_RE     = re.compile(r'[\u0C00-\u0C7F]')   # Telugu
ENGLISH_RE    = re.compile(r'[a-zA-Z]')

# Common Romanised Hindi/Telugu words to detect Hinglish/Tenglish
HINGLISH_WORDS = {
    'yeh', 'ye', 'khabar', 'jhooth', 'jhuth', 'hai', 'he', 'bilkul', 
    'sach', 'sacha', 'sachi', 'samachar', 'aap', 'kaise', 'aur', 'ko', 
    'se', 'ki', 'ka', 'ke', 'kar', 'karo', 'mat', 'bhejo', 'forward', 
    'sab', 'log', 'dosto', 'mitron', 'bhai', 'behan', 'jhut', 'jhuta',
    'karo', 'gaya', 'gayi', 'hua', 'huye', 'hogi', 'hoga'
}
TENGLISH_WORDS = {
    'idi', 'adi', 'varta', 'vartha', 'tappu', 'nizam', 'nijam', 'nammavaddu',
    'pampandi', 'telugu', 'loni', 'gurinchi', 'chepparu', 'avutundi', 'undhi',
    'kuda', 'kani', 'enduku', 'ala', 'ila', 'cheyandi', 'pettandi'
}

def detect_language(text: str) -> str:
    """
    Returns: 'hindi', 'telugu', 'english', or 'other'
    Uses Unicode script detection first (more reliable for short texts),
    then Hinglish/Tenglish heuristics,
    and finally falls back to langdetect for ambiguous cases.
    """
    text = text.strip()
    if not text: return 'english'

    # Count script characters
    deva_count = len(DEVANAGARI_RE.findall(text))
    telu_count = len(TELUGU_RE.findall(text))
    eng_count  = len(ENGLISH_RE.findall(text))
    total      = max(len(text), 1)

    # Script-based detection (works even on 5-word texts)
    if deva_count / total > 0.2: return 'hindi'
    if telu_count / total > 0.2: return 'telugu'

    # Check Romanised script heuristics (Hinglish/Tenglish)
    words = set(re.findall(r'\b\w+\b', text.lower()))
    if words.intersection(HINGLISH_WORDS): return 'hindi'
    if words.intersection(TENGLISH_WORDS): return 'telugu'

    # Mostly Latin script → try langdetect
    if eng_count / total > 0.4:
        try:
            lang = detect(text)
            if lang == 'hi': return 'hindi'    # Hinglish
            if lang == 'te': return 'telugu'   # Tenglish
            if lang == 'en': return 'english'
        except: pass
        return 'english'

    return 'other'


# Test
if __name__ == '__main__':
    tests = [
        ("Scientists confirm vaccine is 95% effective",         "english"),
        ("यह खबर बिल्कुल झूठी है और लोगों को गुमराह करती है",  "hindi"),
        ("ఈ వార్త పూర్తిగా తప్పు మరియు ప్రజలను తప్పుదారి పట్టిస్తుంది", "telugu"),
        ("Yeh khabar bilkul jhooth hai",                         "hindi"),
    ]
    print("Language detection test:")
    for text, expected in tests:
        result = detect_language(text)
        status = '✓' if result == expected else '✗'
        print(f"  {status} [{result:8s}]  {text[:50]}")
