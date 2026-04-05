import os
from data_loader import load_and_combine_data
from eda import check_basic_stats, check_word_count_stats
from features import add_text_features
from visualize import plot_class_distribution, plot_word_count_distribution

def main():
    print("Libraries loaded successfully!")
    print("Starting Phase 1 Analysis...\n")
    
    # 1. Load Data
    # Download from Kaggle first:
    # https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
    print("--- Loading Data ---")
    df = load_and_combine_data(fake_path="fake_news_project/Fake.csv", real_path="fake_news_project/True.csv")
    if df is None:
        return
    
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(df.head())
    
    # 2. Basic EDA
    print("\n--- Basic Statistics ---")
    check_basic_stats(df)
    
    # 3. Visualization: Class Distribution
    print("\n--- Class Distribution ---")
    plot_class_distribution(df)
    
    # 4. Feature Engineering
    print("\n--- Feature Engineering ---")
    df = add_text_features(df)
    
    # 5. Word Count Stats and Visualization
    print("\n--- Word Count Analysis ---")
    check_word_count_stats(df)
    plot_word_count_distribution(df)
    
    print("\nPhase 1 Analysis Complete!")

if __name__ == "__main__":
    main()
