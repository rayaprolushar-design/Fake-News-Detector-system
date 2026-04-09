import re
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER once to save time
analyzer = SentimentIntensityAnalyzer()

def extract_features(texts):
    """
    Extracts 12 semantic and stylistic features from a list or Series of texts.
    Returns a Pandas DataFrame with 12 numeric columns.
    """
    features_list = []
    
    for text in texts:
        text_str = str(text)
        
        # 1. Exclamation marks count
        exclamation_count = text_str.count('!')
        
        # 2. Quotes count
        quote_count = text_str.count('"') + text_str.count("'")
        
        # 3. Question marks count
        question_count = text_str.count('?')
        
        # 4. Word count
        words = text_str.split()
        word_count = len(words)
        
        # 5. Average word length
        avg_word_length = sum(len(w) for w in words) / max(1, word_count)
        
        # 6. All caps words count (words with 2+ uppercase letters)
        all_caps_count = len(re.findall(r'\b[A-Z]{2,}\b', text_str))
        
        # 7. Clickbait score (heuristic combining punctuation + casing)
        clickbait_score = exclamation_count + all_caps_count + question_count
        
        # Sentiment extraction
        sentiment = analyzer.polarity_scores(text_str)
        
        # 8-11. Sentiment scores
        sent_compound = sentiment['compound']
        sent_pos = sentiment['pos']
        sent_neu = sentiment['neu']
        sent_neg = sentiment['neg']
        
        # 12. Digit count (can represent reliance on data/numbers)
        digit_count = sum(c.isdigit() for c in text_str)
        
        features_list.append([
            exclamation_count, quote_count, question_count, word_count,
            avg_word_length, all_caps_count, clickbait_score,
            sent_compound, sent_pos, sent_neu, sent_neg, digit_count
        ])
    
    columns = [
        'exclamation_count', 'quote_count', 'question_count', 'word_count',
        'avg_word_length', 'all_caps_count', 'clickbait_score',
        'sent_compound', 'sent_pos', 'sent_neu', 'sent_neg', 'digit_count'
    ]
    
    return pd.DataFrame(features_list, columns=columns)
