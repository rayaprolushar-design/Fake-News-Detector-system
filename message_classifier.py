# message_classifier.py
# Classifies what kind of message was sent BEFORE running fake news detection
# Prevents the model from analysing things it wasn't designed for

import re

# Patterns that indicate a QUESTION (not a news claim)
QUESTION_PATTERNS = [
    r'\bis\s+(?:the|this|that|it)\b.*\?',
    r'\bwhat\s+is\b',
    r'\bhow\s+(?:do|does|can|to)\b',
    r'\bwhy\s+(?:is|are|did|does)\b',
    r'\bcan\s+you\b',
    r'^is\s+(?:the|this|that)\b',    # starts with "is the..."
    r'\?$',                              # ends with question mark
    r'\bor\s+(?:is|the)\b.*\?',          # "X or Y?" pattern
]

# Words that indicate an ADVERTISEMENT
AD_KEYWORDS = {
    'emi', 'offer', 'discount', 'sale', 'buy now', 'limited time',
    'price', 'deal', 'cashback', 'off', 'shop now', 'order now',
    'advertisement', 'ad', 'promo', 't&c apply', 'terms apply',
    'finance', 'loan', 'interest rate', 'down payment', 'showroom',
    'ex-showroom', 'on road price', 'test drive', 'book now'
}

# Known brands — if user is asking about brand content, it's not fake news
KNOWN_BRANDS = {
    'bmw', 'mercedes', 'audi', 'honda', 'toyota', 'tata',
    'maruti', 'hyundai', 'apple', 'samsung', 'oneplus', 'jio',
    'airtel', 'amazon', 'flipkart', 'swiggy', 'zomato',
    'coca cola', 'pepsi', 'nestle', 'amul', 'britannia'
}

# Words that signal a news CLAIM (things to analyse)
NEWS_SIGNALS = {
    'breaking', 'report', 'confirms', 'says', 'claims', 'alleged',
    'government', 'minister', 'court', 'arrested', 'died', 'killed',
    'election', 'virus', 'vaccine', 'hospital', 'police', 'army',
    'discovered', 'revealed', 'exposed', 'leaked', 'official',
    'study', 'research', 'scientists', 'according to', 'sources say'
}

def classify_message(text: str) -> dict:
    """
    Classify what type of message this is.
    Returns: {
      'type': 'news_claim' | 'question' | 'question_about_ad' | 'advertisement' | 'too_short',
      'confidence': float,
      'reason': str
    }
    """
    text  = text.strip()
    lower = text.lower()
    words = lower.split()
    wc    = len(words)

    # ── Too short to analyse ──────────────────────────
    if wc < 5:
        return {
            'type'      : 'too_short',
            'confidence': 1.0,
            'reason'    : f"Only {wc} words — not enough to analyse as a news claim."
        }

    # ── Question detection ────────────────────────────
    is_question = text.endswith('?')
    for pattern in QUESTION_PATTERNS:
        if re.search(pattern, lower):
            is_question = True
            break

    # ── Brand / advertisement detection ─────────────
    brand_hits = sum(1 for b in KNOWN_BRANDS if b in lower)
    ad_hits    = sum(1 for a in AD_KEYWORDS if a in lower)

    # ── News signal detection ────────────────────────
    news_hits  = sum(1 for n in NEWS_SIGNALS if n in lower)

    # ── Decision logic ───────────────────────────────

    # Question about an ad/brand → not a news claim
    if is_question and (brand_hits > 0 or ad_hits > 0):
        return {
            'type'      : 'question_about_ad',
            'confidence': 0.95,
            'reason'    : "User is asking a question about an advertisement or brand."
        }

    # Pure question with no news signals
    if is_question and news_hits == 0:
        return {
            'type'      : 'question',
            'confidence': 0.90,
            'reason'    : "This looks like a question, not a news claim."
        }

    # Advertisement content
    if ad_hits >= 2 and news_hits == 0:
        return {
            'type'      : 'advertisement',
            'confidence': 0.88,
            'reason'    : "This looks like an advertisement, not a news claim."
        }

    # Has clear news signals → analyse it
    if news_hits >= 1:
        return {
            'type'      : 'news_claim',
            'confidence': 0.85,
            'reason'    : f"{news_hits} news signal(s) detected."
        }

    # Short question-like structure with no news signals
    if wc < 15 and news_hits == 0 and is_question:
        return {
            'type'      : 'question',
            'confidence': 0.80,
            'reason'    : "Short text with no news signals."
        }

    # Default — treat as news claim and let the model decide
    return {
        'type'      : 'news_claim',
        'confidence': 0.60,
        'reason'    : "Treating as a news claim for analysis."
    }


# Test — including the BMW case
if __name__ == '__main__':
    tests = [
        "Is the advertisement fake or the advertisement is real",
        "SHOCKING!! Vaccine causes 5G connection!! Forward this!!",
        "BMW X1 at EMI of 29999 with 5.75% ROI and road tax benefits",
        "Scientists confirm new diabetes cure discovered in India",
        "hi",
        "What is the price of gold today?",
    ]
    for t in tests:
        r = classify_message(t)
        print(f"[{r['type']:22s}]  {t[:55]}")
