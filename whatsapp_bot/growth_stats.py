import os
import pandas as pd

# Load usage log from script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'usage_log.csv')

if not os.path.exists(csv_path):
    print("=== VerifyAI Growth Stats ===")
    print("No usage log found yet. Interact with the bot to generate data.")
    exit(0)

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['timestamp']).dt.date

print("=== VerifyAI Growth Stats ===")
print(f"Total messages   : {len(df)}")
print(f"Unique users     : {df['user_hash'].nunique()}")
print(f"Share commands   : {(df['message_type']=='share_command').sum()}")
print(f"\nBy message type:")
print(df['message_type'].value_counts())
print(f"\nMessages per day:")
print(df.groupby('date').size())
