import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


def plot_confusion_matrix(y_test, preds, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Fake', 'Predicted Real'],
                yticklabels=['Actual Fake', 'Actual Real'])

    ax.set_title('Confusion Matrix — Logistic Regression')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    # Read the numbers
    tn, fp, fn, tp = cm.ravel()
    print(f"Correctly called FAKE : {tn}")
    print(f"Correctly called REAL : {tp}")
    print(f"Fake called as Real   : {fp}  ← dangerous!")
    print(f"Real called as Fake   : {fn}")


def print_classification_report(y_test, preds):
    print("\n=== Logistic Regression — Full Report ===")
    print(classification_report(y_test, preds, target_names=['Fake', 'Real']))


def plot_roc_curve(model, X_test, y_test, save_path='roc_curve.png'):
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"AUC-ROC Score: {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_test, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#534AB7', lw=2,
            label=f'Logistic Regression (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Random guess')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#534AB7')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_top_words(model, tfidf, save_path='top_words.png'):
    feature_names = np.array(tfidf.get_feature_names_out())
    # Add the 12 style features
    style_features = [
        'exclamation_count', 'quote_count', 'question_count', 'word_count',
        'avg_word_length', 'all_caps_count', 'clickbait_score',
        'sent_compound', 'sent_pos', 'sent_neu', 'sent_neg', 'digit_count'
    ]
    feature_names = np.concatenate([feature_names, style_features])
    
    coefs = model.coef_[0]  # positive = real, negative = fake

    top_n = 15
    top_fake_idx = coefs.argsort()[:top_n]       # most negative = fake
    top_real_idx = coefs.argsort()[-top_n:][::-1]  # most positive = real

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fake words
    axes[0].barh(feature_names[top_fake_idx],
                 np.abs(coefs[top_fake_idx]),
                 color='#E24B4A')
    axes[0].set_title('Top 15 words/features pointing to FAKE')
    axes[0].invert_yaxis()

    # Real words
    axes[1].barh(feature_names[top_real_idx],
                 coefs[top_real_idx],
                 color='#1D9E75')
    axes[1].set_title('Top 15 words/features pointing to REAL')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_rf_importances(model, tfidf, save_path='rf_importances.png'):
    print("\nRanking features with Random Forest...")
    feature_names = np.array(tfidf.get_feature_names_out())
    style_features = [
        'exclamation_count', 'quote_count', 'question_count', 'word_count',
        'avg_word_length', 'all_caps_count', 'clickbait_score',
        'sent_compound', 'sent_pos', 'sent_neu', 'sent_neg', 'digit_count'
    ]
    feature_names = np.concatenate([feature_names, style_features])
    
    importances = model.feature_importances_
    
    # Get top 20
    top_n = 20
    top_idx = importances.argsort()[-top_n:][::-1]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_names[top_idx], importances[top_idx], color='#2C7BB6')
    ax.set_title('Top 20 Features (Random Forest)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print("\n--- TOP 10 STRONGEST FEATURES ---")
    for i, idx in enumerate(top_idx[:10]):
        val = importances[idx]
        name = feature_names[idx]
        marker = "⭐" if name in ['clickbait_score', 'sent_compound'] else "  "
        print(f"{i+1:2}. {marker} {name:<20} ({val:.4f})")
