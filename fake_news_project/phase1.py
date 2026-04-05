import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Makes plots look clean
plt.style.use('seaborn-v0_8-whitegrid')
print("Libraries loaded successfully!")

# Download from Kaggle first:
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
# You'll get two files: Fake.csv and True.csv
# Put them in the same folder as this script

fake_df = pd.read_csv('Fake.csv')
real_df = pd.read_csv('True.csv')

# Add a label column: 0 = fake, 1 = real
fake_df['label'] = 0
real_df['label']  = 1

# Combine into one dataframe
df = pd.concat([fake_df, real_df], ignore_index=True)

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
df.head()

# Always check this before doing anything else!
print("=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Data Types ===")
print(df.dtypes)

print("\n=== Shape ===")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# Check how many fake vs real articles we have
counts = df['label'].value_counts()
print("Fake articles: ", counts[0])
print("Real articles: ", counts[1])

# Plot it
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['Fake', 'Real'], [counts[0], counts[1]],
       color=['#E24B4A', '#1D9E75'])
ax.set_title('Fake vs Real article count')
ax.set_ylabel('Number of articles')
plt.tight_layout()
plt.savefig('class_distribution.png', dpi=150)
plt.show()

# Add word count column — useful feature later!
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
df['title_len']  = df['title'].apply(lambda x: len(str(x).split()))

print("=== Fake article word count ===")
print(df[df['label']==0]['word_count'].describe())
print("\n=== Real article word count ===")
print(df[df['label']==1]['word_count'].describe())

# Side-by-side histogram
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df[df['label']==0]['word_count'],
             bins=50, color='#E24B4A', alpha=0.7)
axes[0].set_title('Fake — word count distribution')

axes[1].hist(df[df['label']==1]['word_count'],
             bins=50, color='#1D9E75', alpha=0.7)
axes[1].set_title('Real — word count distribution')

plt.tight_layout()
plt.savefig('word_count_dist.png', dpi=150)
plt.show()

# Save for Phase 2
print("Saving news_data_phase1.csv...")
df.to_csv('news_data_phase1.csv', index=False)
print("Done!")
