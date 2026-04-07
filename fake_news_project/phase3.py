import pickle
from model_training import load_data, train_logistic_regression, train_naive_bayes
from model_evaluation import (
    plot_confusion_matrix, print_classification_report,
    plot_roc_curve, plot_top_words
)
from predictions import predict_news


def main():
    print("Starting Phase 3...")

    # Load Data
    try:
        X_train, X_test, y_train, y_test = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure you have run phase2.py so the data files exist.")
        return

    print(f"X_train: {X_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_train: {y_train.shape}, y_test: {y_test.shape}\n")

    # Train Models
    lr_model, lr_preds, lr_acc = train_logistic_regression(
        X_train, y_train, X_test, y_test)
    print(f"Logistic Regression accuracy: {lr_acc * 100:.2f}%\n")

    nb_model, nb_preds, nb_acc = train_naive_bayes(
        X_train, y_train, X_test, y_test)
    print(f"Naive Bayes accuracy: {nb_acc * 100:.2f}%\n")

    # Quick comparison
    print("--- Model Comparison ---")
    print(f"Logistic Regression : {lr_acc * 100:.2f}%")
    print(f"Naive Bayes         : {nb_acc * 100:.2f}%")
    winner = 'Logistic Regression' if lr_acc > nb_acc else 'Naive Bayes'
    print(f"Winner              : {winner}\n")

    # Evaluation on Logistic Regression
    print("--- Evaluation: Logistic Regression ---")
    plot_confusion_matrix(y_test, lr_preds)
    print_classification_report(y_test, lr_preds)
    plot_roc_curve(lr_model, X_test, y_test)

    # Load TF-IDF vocabulary for feature importance and predictions
    print("\nLoading TF-IDF Vectorizer for Feature Analysis...")
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)

    # Feature Importance
    print("--- Feature Importance ---")
    plot_top_words(lr_model, tfidf)

    # Predictions
    print("\n--- Try it out! ---")
    predict_news(
        "Scientists confirm new vaccine is 95% effective",
        lr_model,
        tfidf)
    predict_news(
        "SHOCKING: Government hiding alien contact since 1947!!",
        lr_model,
        tfidf)
    predict_news(
        "Federal Reserve holds interest rates steady",
        lr_model,
        tfidf)
    predict_news(
        "Taylor Swift announces surprise album release",
        lr_model,
        tfidf)

    # Save model
    with open('lr_model.pkl', 'wb') as f:
        pickle.dump(lr_model, f)

    print("\nSaved lr_model.pkl")
    print("tfidf_vectorizer.pkl already saved from Phase 2")
    print("\nFiles ready for Phase 4:")
    print("  lr_model.pkl")
    print("  tfidf_vectorizer.pkl")
    print(
        f"\nPhase 3 complete! Your model hits {max(lr_acc, nb_acc) * 100:.2f}% accuracy.")


if __name__ == "__main__":
    main()
