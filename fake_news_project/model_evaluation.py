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
    plt.show()

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
    ax.plot([0,1], [0,1], '--', color='gray', label='Random guess')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#534AB7')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()

def plot_top_words(model, tfidf, save_path='top_words.png'):
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = model.coef_[0]  # positive = real, negative = fake

    top_n = 15
    top_fake_idx = coefs.argsort()[:top_n]       # most negative = fake
    top_real_idx = coefs.argsort()[-top_n:][::-1] # most positive = real

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fake words
    axes[0].barh(feature_names[top_fake_idx],
                  np.abs(coefs[top_fake_idx]),
                  color='#E24B4A')
    axes[0].set_title('Top 15 words pointing to FAKE')
    axes[0].invert_yaxis()

    # Real words
    axes[1].barh(feature_names[top_real_idx],
                  coefs[top_real_idx],
                  color='#1D9E75')
    axes[1].set_title('Top 15 words pointing to REAL')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
