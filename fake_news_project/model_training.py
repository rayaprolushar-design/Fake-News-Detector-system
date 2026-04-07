import scipy.sparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def load_data():
    """Loads train and test arrays and labels from disk."""
    X_train = scipy.sparse.load_npz('X_train.npz')
    X_test = scipy.sparse.load_npz('X_test.npz')
    y_train = pd.read_csv('y_train.csv').squeeze()
    y_test = pd.read_csv('y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test


def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Trains Logistic Regression and returns model, predictions, and accuracy."""
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,      # enough iterations to converge
        C=1.0,              # regularization strength (default is good)
        solver='lbfgs',
        random_state=42
    )
    lr_model.fit(X_train, y_train)

    lr_preds = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_preds)
    return lr_model, lr_preds, lr_acc


def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Trains Naive Bayes and returns model, predictions, and accuracy."""
    print("Training Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)  # alpha = smoothing
    nb_model.fit(X_train, y_train)

    nb_preds = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_preds)
    return nb_model, nb_preds, nb_acc
