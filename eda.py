def check_basic_stats(df):
    """Prints basic exploratory data analysis statistics."""
    # Always check this before doing anything else!
    print("=== Missing Values ===")
    print(df.isnull().sum())

    print("\n=== Data Types ===")
    print(df.dtypes)

    print("\n=== Shape ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

def check_word_count_stats(df):
    """Prints summary statistics for word count in fake and real articles."""
    print("=== Fake article word count ===")
    print(df[df['label']==0]['word_count'].describe())
    
    print("\n=== Real article word count ===")
    print(df[df['label']==1]['word_count'].describe())