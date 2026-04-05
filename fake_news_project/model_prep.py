import pickle
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_and_save_features(df):
    """Splits data, builds TF-IDF, and saves matrices and models."""
    # Split into train (80%) and test (20%) BEFORE fitting TF-IDF
    X = df['combined']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # keeps fake/real ratio same in both splits
    )

    print(f"Training set: {len(X_train)} articles")
    print(f"Test set:     {len(X_test)} articles")

    # Build TF-IDF — only fit on training data!
    tfidf = TfidfVectorizer(
        max_features=10000,  # use top 10,000 most useful words
        ngram_range=(1, 2),  # single words AND two-word pairs
        min_df=5             # ignore words appearing in fewer than 5 articles
    )

    X_train_tfidf = tfidf.fit_transform(X_train)  # learn + transform
    X_test_tfidf  = tfidf.transform(X_test)       # transform only (no fitting!)

    print(f"\nTF-IDF matrix shape: {X_train_tfidf.shape}")
    print("(rows = articles, columns = word features)")

    # Save the TF-IDF model
    with open('tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)

    # Save train/test splits
    scipy.sparse.save_npz('X_train.npz', X_train_tfidf)
    scipy.sparse.save_npz('X_test.npz',  X_test_tfidf)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv',  index=False)

    print("Saved:")
    print("  tfidf_vectorizer.pkl")
    print("  X_train.npz / X_test.npz")
    print("  y_train.csv / y_test.csv")
