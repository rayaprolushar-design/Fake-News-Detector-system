from text_processing import clean_text

def predict_news(headline, model, tfidf):
    """Predicts if a headline is fake or real using a trained model."""
    cleaned = clean_text(headline)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    confidence = model.predict_proba(vectorized)[0]
    
    label = "REAL" if prediction == 1 else "FAKE"
    conf_pct = max(confidence) * 100
    
    print(f"Headline : {headline}")
    print(f"Result   : {label}  ({conf_pct:.1f}% confident)\n")
