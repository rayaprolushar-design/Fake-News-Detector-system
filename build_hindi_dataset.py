# build_hindi_dataset.py
# Assembles Hindi + Telugu fake news training data from free sources

import os
import pandas as pd
from datasets import load_dataset
from data_loader import load_and_combine_data
from lang_detector import detect_language

dfs = []

# ── Source 1: Hindi Fake News dataset (Hugging Face) ──
print("Loading Hindi Fake News dataset...")
try:
    ds = load_dataset("rahular/hindi-fake-news")
    df1 = ds['train'].to_pandas()
    df1 = df1.rename(columns={'headline': 'text'})[['text', 'label']]
    df1['lang'] = 'hindi'
    dfs.append(df1)
    print(f"  ✓ {len(df1)} Hindi articles")
except Exception as e:
    print(f"  ✗ {e} — trying fallback")

# ── Source 2: IFND dataset ─────────────────────────
print("Loading IFND dataset...")
try:
    ds2 = load_dataset("Keelback/ifnd-fake-news")
    df2 = ds2['train'].to_pandas()
    # Keep only Hindi/Telugu rows
    df2['lang'] = df2['text'].apply(detect_language)
    df2 = df2[df2['lang'].isin(['hindi', 'telugu'])]
    dfs.append(df2[['text', 'label', 'lang']])
    print(f"  ✓ {len(df2)} Hindi/Telugu articles")
except Exception as e:
    print(f"  ✗ {e}")

# ── Source 3: Translate a subset of English data ──
# Use our existing English dataset + translate 2000 rows to Hindi
# This is called cross-lingual transfer and dramatically boosts performance
print("Building translated subset...")
try:
    from deep_translator import GoogleTranslator
    
    # Load the English dataset using our existing data loader
    fake_csv = os.path.join("fake_news_project", "Fake.csv")
    true_csv = os.path.join("fake_news_project", "True.csv")
    eng_df = load_and_combine_data(fake_path=fake_csv, real_path=true_csv)
    
    sample = eng_df.sample(2000, random_state=42)  # translate 2000 rows

    translated = []
    print("Starting translation process (this may take a few minutes)...")
    for idx, (i, row) in enumerate(sample.iterrows()):
        try:
            hi_text = GoogleTranslator(source='en', target='hi').translate(
                row['text'][:500]
            )
            translated.append({'text': hi_text, 'label': row['label'], 'lang': 'hindi'})
        except Exception as te:
            continue
        if (idx + 1) % 200 == 0:
            print(f"  Translated {idx + 1}/2000...")

    df3 = pd.DataFrame(translated)
    dfs.append(df3)
    print(f"  ✓ {len(df3)} translated articles")
except Exception as e:
    print(f"  ✗ Translation failed: {e} — skip and continue")

# ── Combine, clean, save ───────────────────────────
if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=['text','label'])
    combined = combined.drop_duplicates(subset=['text'])
    combined.to_csv('hindi_telugu_dataset.csv', index=False)
    print(f"\n✓ Final dataset: {len(combined)} rows")
    if 'lang' in combined.columns:
        print(combined['lang'].value_counts())
else:
    print("Warning: No datasets loaded. Please check internet connection or database states.")
