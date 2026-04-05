import re
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Ensure stopwords are downloaded
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Cleans text by lowercasing, removing URLs, punctuation, and stopwords."""
    # Step 1: lowercase everything
    text = str(text).lower()

    # Step 2: remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 3: remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Step 4: remove stopwords
    words = text.split()
    words = [w for w in words if w not in STOPWORDS]

    # Step 5: rejoin into a clean string
    return ' '.join(words)

def apply_text_cleaning(df):
    """Applies text cleaning to title and text, returning a combined column."""
    print("Cleaning text... please wait (this may take 30-60 seconds)")
    df['clean_text'] = df['text'].apply(clean_text)
    df['clean_title'] = df['title'].apply(clean_text)

    # Combine title + text for stronger signal
    df['combined'] = df['clean_title'] + ' ' + df['clean_text']
    
    print("Done! Sample cleaned article:")
    print(df['combined'].iloc[0][:200])
    return df
