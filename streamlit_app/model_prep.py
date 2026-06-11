import pickle
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from features import extract_features

def prepare_and_save_features(df):
    """Splits data, builds TF-IDF, adds style features, and saves matrices and models."""
    print("Extracting 12 style and sentiment features (this might take a minute)...")
    style_features_df = extract_features(df['text'])
    
    # We need to split text and style features identically.
    X_text = df['combined']
    X_style = style_features_df
    y = df['label']
    
    X_text_train, X_text_test, X_style_train, X_style_test, y_train, y_test = train_test_split(
        X_text, X_style, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Training set: {len(X_text_train)} articles")
    print(f"Test set:     {len(X_text_test)} articles")

    # Build TF-IDF
    print("Fitting TF-IDF Vectorizer with 15k features...")
    tfidf = TfidfVectorizer(
        max_features=15000,
        ngram_range=(1, 2),
        min_df=5,
        sublinear_tf=True
    )

    X_train_tfidf = tfidf.fit_transform(X_text_train)
    X_test_tfidf = tfidf.transform(X_text_test)

    # Scale the style features
    print("Scaling style features...")
    scaler = StandardScaler()
    X_train_style_scaled = scaler.fit_transform(X_style_train)
    X_test_style_scaled = scaler.transform(X_style_test)
    
    # Convert scaled features to sparse matrix
    X_train_style_sparse = scipy.sparse.csr_matrix(X_train_style_scaled)
    X_test_style_sparse = scipy.sparse.csr_matrix(X_test_style_scaled)
    
    # FUSION trick
    print("Fusing features (TF-IDF + Style)...")
    X_train_fused = scipy.sparse.hstack([X_train_tfidf, X_train_style_sparse])
    X_test_fused = scipy.sparse.hstack([X_test_tfidf, X_test_style_sparse])

    print(f"\nFinal FUSED Matrix Shape (text+style): {X_train_fused.shape}")

    # Save everything
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
        
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    scipy.sparse.save_npz('X_train.npz', X_train_fused)
    scipy.sparse.save_npz('X_test.npz', X_test_fused)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)

    print("Saved:")
    print("  tfidf_vectorizer.pkl")
    print("  scaler.pkl")
    print("  X_train.npz / X_test.npz")
    print("  y_train.csv / y_test.csv")
