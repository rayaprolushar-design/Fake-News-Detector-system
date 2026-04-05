def add_text_features(df):
    """Adds word count and title length features to the dataframe."""
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    df['title_len']  = df['title'].apply(lambda x: len(str(x).split()))
    return df
