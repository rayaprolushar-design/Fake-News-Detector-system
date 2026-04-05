import pandas as pd

def load_and_combine_data(fake_path='Fake.csv', real_path='True.csv'):
    """Loads Fake and True news datasets and combines them."""
    try:
        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)
        
        # Add a label column: 0 = fake, 1 = real
        fake_df['label'] = 0
        real_df['label'] = 1
        
        # Combine into one dataframe
        df = pd.concat([fake_df, real_df], ignore_index=True)
        return df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure Fake.csv and True.csv are in the same directory.")
        return None