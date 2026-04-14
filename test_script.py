import pandas as pd
from fake_news_project.model_prep import prepare_and_save_features
from fake_news_project.model_training import train_logistic_regression, train_naive_bayes
from fake_news_project.plot_wordclouds import generate_wordclouds
from features import add_text_features

# Fake data
df = pd.DataFrame({
    'title': ['hello world', 'fake news is bad'],
    'text': ['this is some text', 'and another one'],
    'combined': ['hello world this is some text', 'fake news is bad and another one'],
    'label': [1, 0]
})

print("Testing features.py...")
df = add_text_features(df)
print("Features added successfully.")

print("Testing plot_wordclouds...")
# Just don't plt.show
import matplotlib.pyplot as plt
plt.show = lambda: None
generate_wordclouds(df, 'test_wc.png')

print("Testing model_prep...")
prepare_and_save_features(df)

print("Testing model_training...")
import scipy.sparse
import pandas as pd
from fake_news_project.model_training import load_data
X_train, X_test, y_train, y_test = load_data()
train_logistic_regression(X_train, y_train, X_test, y_test)
train_naive_bayes(X_train, y_train, X_test, y_test)

print("All done!")
