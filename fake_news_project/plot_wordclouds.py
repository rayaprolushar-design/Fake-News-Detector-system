import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_wordclouds(df, save_path='wordclouds.png'):
    """Generates and saves wordclouds for Fake and Real news articles."""
    # Combine all fake text into one big string
    fake_text = ' '.join(df[df['label'] == 0]['combined'])
    real_text = ' '.join(df[df['label'] == 1]['combined'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Fake news word cloud
    wc_fake = WordCloud(width=600, height=400,
                        background_color='white',
                        colormap='Reds',
                        max_words=80).generate(fake_text)

    # Real news word cloud
    wc_real = WordCloud(width=600, height=400,
                        background_color='white',
                        colormap='Greens',
                        max_words=80).generate(real_text)

    axes[0].imshow(wc_fake, interpolation='bilinear')
    axes[0].axis('off')
    axes[0].set_title('Top words in FAKE articles', fontsize=13)

    axes[1].imshow(wc_real, interpolation='bilinear')
    axes[1].axis('off')
    axes[1].set_title('Top words in REAL articles', fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Saved {save_path}")
