# trend_tracker.py
# Tracks which claims are being checked most — detects viral misinformation

import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
import re
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(BASE_DIR, 'analysis_log.csv')

STOPWORDS = {
    'the','is','a','of','in','and','to','it','that',
    'this','was','for','on','are','with','as','be','by',
    'an','at','not','or','but','से','है','का','की','के'
}

def get_trending_fake_claims(days: int = 7) -> dict:
    """
    Analyses analysis_log.csv to find:
    1. Most-checked claims (viral topics)
    2. Most common fake keywords
    3. Language breakdown
    """
    try:
        if not os.path.exists(LOG_FILE):
            return {}
        df = pd.read_csv(LOG_FILE)
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        cutoff = (datetime.now() - timedelta(days=days)).date()
        df = df[df['date'] >= cutoff]
        fake_df = df[df['verdict'].isin(['FAKE', 'LIKELY FAKE'])]

        # Extract trending keywords from fake news
        all_words = []
        for text in fake_df['text_preview'].dropna():
            words = re.findall(r'\b[a-zA-Z\u0900-\u097F\u0C00-\u0C7F]{4,}\b', text.lower())
            all_words.extend([w for w in words if w not in STOPWORDS])

        top_keywords = Counter(all_words).most_common(8)

        return {
            'total_fake'     : len(fake_df),
            'total_checks'   : len(df),
            'fake_rate'      : len(fake_df) / max(len(df), 1) * 100,
            'top_keywords'   : top_keywords,
            'lang_breakdown' : df['language'].value_counts().to_dict(),
            'days'           : days
        }
    except Exception as e:
        print(f"Error in trend calculation: {e}")
        return {}


def fmt_trends_message() -> str:
    """WhatsApp reply for /trends command."""
    data = get_trending_fake_claims(days=7)
    if not data:
        return "📊 Not enough data yet — check back after more people use the bot!"

    keywords_str = " · ".join([f"{w} ({c})" for w, c in data['top_keywords'][:5]])
    lang = data['lang_breakdown']

    return f"""📊 *VerifyAI — 7 Day Trend Report*
━━━━━━━━━━━━━━━━━━━━
🔍 Total checks  : {data['total_checks']}
❌ Fake detected : {data['total_fake']} ({data['fake_rate']:.0f}%)

🔥 *Trending fake keywords:*
{keywords_str}

🌐 *By language:*
  English : {lang.get('english', 0)}
  Hindi   : {lang.get('hindi', 0)}
  Telugu  : {lang.get('telugu', 0)}
━━━━━━━━━━━━━━━━━━━━
_VerifyAI Intelligence · Updated daily_"""
