import pandas as pd
from text_processing import clean_text, apply_text_cleaning
from plot_wordclouds import generate_wordclouds
from model_prep import prepare_and_save_features


def main():
    print("All imports successful! Starting Phase 2...")

    # Load the file we saved at the end of Phase 1
    try:
        df = pd.read_csv('news_data_phase1.csv')
        print(f"Loaded {len(df)} rows")
        print(df.head(3))
    except FileNotFoundError:
        print("Error: news_data_phase1.csv not found.")
        print("Please ensure you have run phase1.py completely so it saves the CSV.")
        return

    # Test the cleaner on one example
    sample = "BREAKING: The FBI has opened an investigation!! Visit http://news.com"
    print("\nTesting cleaner on sample text:")
    print("Before:", sample)
    print("After: ", clean_text(sample))

    # 1. Process Text
    print("\n--- Text Processing ---")
    df = apply_text_cleaning(df)

    # 2. Generate Wordclouds
    print("\n--- Generating Wordclouds ---")
    generate_wordclouds(df)

    # 3. Model Preparation (TF-IDF + Splits)
    print("\n--- Model Preparation ---")
    prepare_and_save_features(df)

    print("\nPhase 2 complete!")


if __name__ == "__main__":
    main()
