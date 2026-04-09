from text_processing import clean_text
from features import extract_features
import scipy.sparse

def predict_news(headline, model, tfidf, scaler):
    """Predicts if a headline is fake or real using a trained model."""
    cleaned = clean_text(headline)
    text_vector = tfidf.transform([cleaned])
    
    style_df = extract_features([headline])
    style_scaled = scaler.transform(style_df)
    style_sparse = scipy.sparse.csr_matrix(style_scaled)
    
    fused_features = scipy.sparse.hstack([text_vector, style_sparse])
    
    prediction = model.predict(fused_features)[0]
    confidence = model.predict_proba(fused_features)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    conf_pct = max(confidence) * 100

    print(f"Headline : {headline}")
    print(f"Result   : {label}  ({conf_pct:.1f}% confident)\n")
