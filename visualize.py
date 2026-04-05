import matplotlib.pyplot as plt

def apply_plot_style():
    """Applies clean plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')

def plot_class_distribution(df, save_path='class_distribution.png'):
    """Plots the distribution of fake vs real articles."""
    apply_plot_style()
    counts = df['label'].value_counts()
    
    print("Fake articles: ", counts.get(0, 0))
    print("Real articles: ", counts.get(1, 0))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(['Fake', 'Real'], [counts.get(0, 0), counts.get(1, 0)],
           color=['#E24B4A', '#1D9E75'])
    ax.set_title('Fake vs Real article count')
    ax.set_ylabel('Number of articles')
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()

def plot_word_count_distribution(df, save_path='word_count_dist.png'):
    """Plots a side-by-side histogram of word counts for fake and real articles."""
    apply_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df[df['label']==0]['word_count'],
                 bins=50, color='#E24B4A', alpha=0.7)
    axes[0].set_title('Fake — word count distribution')

    axes[1].hist(df[df['label']==1]['word_count'],
                 bins=50, color='#1D9E75', alpha=0.7)
    axes[1].set_title('Real — word count distribution')

    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.show()
