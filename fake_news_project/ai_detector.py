import re, math, nltk
import numpy as np
from collections import Counter
import textstat

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('cmudict', quiet=True)

# Words AI models overuse — compiled from research papers
# on GPT/Claude/Gemini output patterns
AI_OVERUSED = {
    'delve', 'leverage', 'crucial', 'furthermore', 'moreover',
    'however', 'notably', 'significant', 'straightforward',
    'subsequently', 'consequently', 'utilize', 'facilitate',
    'implement', 'comprehensive', 'pivotal', 'imperative',
    'multifaceted', 'undeniably', 'invaluable', 'vibrant',
    'robust', 'streamline', 'cutting-edge', 'game-changer',
    'paradigm', 'synergy', 'holistic', 'dive into', 'it is worth',
    'in conclusion', 'in summary', 'to summarize',
    'it is important', 'one must', 'this allows', 'this ensures'
}

def score_perplexity(text: str) -> float:
    """
    Estimate perplexity using unigram entropy.
    Low entropy = predictable = AI-like.
    Returns a 0-1 score where 0 = very AI-like, 1 = very human-like.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 5: return 0.5

    counts = Counter(words)
    total  = len(words)
    probs  = [c/total for c in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs)

    # Normalise: human text entropy ~3.5-4.5, AI ~2.5-3.5
    score = min(max((entropy - 2.0) / 3.0, 0), 1)
    return round(score, 3)

def score_burstiness(text: str) -> float:
    """
    Humans vary sentence length a lot (high burstiness).
    AI writes uniformly medium-length sentences (low burstiness).
    Returns 0-1 where 1 = human-like burstiness.
    """
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2: return 0.5

    lengths = [len(s.split()) for s in sentences]
    mean    = np.mean(lengths)
    std     = np.std(lengths)
    cv      = std / max(mean, 1)  # coefficient of variation

    # Human CV ~0.5-1.5, AI CV ~0.1-0.4
    score = min(max(cv / 1.2, 0), 1)
    return round(score, 3)

def score_ai_vocabulary(text: str) -> float:
    """
    Count AI-overused words.
    Returns 0-1 where 0 = clean human vocab, 1 = very AI-like vocab.
    """
    lower = text.lower()
    hits  = sum(1 for w in AI_OVERUSED if w in lower)
    # Normalise — 3+ hits is very AI-like
    return round(min(hits / 3.0, 1.0), 3)

def score_repetition(text: str) -> float:
    """
    AI tends to repeat the same phrases and sentence starters.
    Returns 0-1 where 0 = varied (human), 1 = repetitive (AI).
    """
    sentences = [s.strip().lower() for s in
                 re.split(r'[.!?]+', text) if s.strip()]
    if len(sentences) < 2: return 0.0

    # Check first 3 words of each sentence (AI loves same starters)
    starters = [' '.join(s.split()[:3]) for s in sentences if len(s.split()) >= 3]
    if not starters: return 0.0

    unique_ratio   = len(set(starters)) / len(starters)
    repetition_score = 1 - unique_ratio
    return round(repetition_score, 3)

def get_flagged_words(text: str) -> list:
    """Return which AI-overused words appear in the text."""
    lower = text.lower()
    return [w for w in AI_OVERUSED if w in lower]

def detect_ai_text(text: str) -> dict:
    """
    Main function. Returns full analysis dict.
    Works reliably on text as short as 10 words.
    """
    text = text.strip()
    word_count = len(text.split())

    # --- Compute 4 signals ---
    perplexity_score  = score_perplexity(text)    # high = human
    burstiness_score  = score_burstiness(text)    # high = human
    vocab_ai_score    = score_ai_vocabulary(text)  # high = AI
    repetition_score  = score_repetition(text)    # high = AI

    # Readability — AI writes at a higher grade level
    readability = textstat.flesch_kincaid_grade(text)
    readability_score = min(max((readability - 8) / 10, 0), 1)  # high = AI

    # Adjust weights for short text (< 30 words)
    # Fewer signals are reliable → lean on vocab and readability
    if word_count < 30:
        ai_score = (
            (1 - perplexity_score) * 0.15 +
            (1 - burstiness_score) * 0.10 +
            vocab_ai_score          * 0.50 +
            repetition_score        * 0.10 +
            readability_score       * 0.15
        )
    else:
        ai_score = (
            (1 - perplexity_score) * 0.25 +
            (1 - burstiness_score) * 0.25 +
            vocab_ai_score          * 0.25 +
            repetition_score        * 0.15 +
            readability_score       * 0.10
        )

    ai_score = round(ai_score * 100, 1)

    # Verdict
    if   ai_score >= 75: verdict, note = "AI GENERATED",   "Strong signals of AI writing."
    elif ai_score >= 55: verdict, note = "LIKELY AI",      "Leans AI — check flagged words."
    elif ai_score >= 40: verdict, note = "UNCERTAIN",      "Mixed signals — could be either."
    elif ai_score >= 20: verdict, note = "LIKELY HUMAN",   "Leans human — natural patterns."
    else:               verdict, note = "HUMAN WRITTEN",  "Strong signals of human writing."

    return {
        'ai_score'        : ai_score,
        'verdict'         : verdict,
        'note'            : note,
        'word_count'      : word_count,
        'signals': {
            'perplexity' : round((1 - perplexity_score) * 100, 1),
            'burstiness': round((1 - burstiness_score) * 100, 1),
            'ai_vocab'  : round(vocab_ai_score * 100, 1),
            'repetition': round(repetition_score * 100, 1),
            'readability': round(readability_score * 100, 1),
        },
        'flagged_words'   : get_flagged_words(text),
        'short_text_mode' : word_count < 30
    }

# Quick test
if __name__ == '__main__':
    ai_text = ("It is crucial to leverage comprehensive strategies that "
               "facilitate robust outcomes. Furthermore, this ensures "
               "streamlined implementation of pivotal paradigms.")

    human_text = ("I tried the new café near my house. Terrible coffee. "
                  "But the owner was nice so I gave it 3 stars anyway lol")

    short_text = "This is a crucial and comprehensive update."

    for t in [ai_text, human_text, short_text]:
        r = detect_ai_text(t)
        print(f"[{r['verdict']:15s}] ({r['ai_score']:5.1f}% AI)  words={r['word_count']}")
        print(f"  Flagged: {r['flagged_words']}")
        print()
