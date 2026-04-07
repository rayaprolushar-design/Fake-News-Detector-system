import pandas as pd
from datasets import load_dataset

def load_and_combine_data(fake_path=None, real_path=None):
    """Loads datasets from Hugging Face and combines them."""
    print("Downloading GossipCop dataset...")
    # Load from Hugging Face — no account needed
    gossip = load_dataset("newsmediabias/fake-news-detection-GossipCop")
    gossip_df = gossip['train'].to_pandas()
    
    print(f"Columns: {list(gossip_df.columns)}")
    print(f"Rows: {len(gossip_df)}")
    
    print("Downloading LIAR dataset...")
    liar_raw = load_dataset("liar")
    liar_train = liar_raw['train'].to_pandas()
    liar_val   = liar_raw['validation'].to_pandas()
    liar_test  = liar_raw['test'].to_pandas()
    
    liar_df = pd.concat([liar_train, liar_val, liar_test], ignore_index=True)
    
    print(f"Columns: {list(liar_df.columns)}")
    print(f"Rows: {len(liar_df)}")
    print("Label values:", liar_df['label'].unique())
    
    # Merge and Label fixing
    liar_df['label'] = liar_df['label'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
    
    # Schemas
    if 'text' not in liar_df.columns and 'statement' in liar_df.columns:
        liar_df['text'] = liar_df['statement']
    if 'title' not in liar_df.columns:
        liar_df['title'] = ''
        
    if 'text' not in gossip_df.columns:
        gossip_df['text'] = gossip_df.get('title', '')
    if 'title' not in gossip_df.columns:
        gossip_df['title'] = ''
        
    gossip_df['label'] = pd.to_numeric(gossip_df['label'], errors='coerce').fillna(0).astype(int)
    
    df = pd.concat([
        gossip_df[['title', 'text', 'label']], 
        liar_df[['title', 'text', 'label']]
    ], ignore_index=True)
    
    # Balance Data (40k fake, 39k real)
    fake_df = df[df['label'] == 0]
    real_df = df[df['label'] == 1]
    
    n_fake = min(len(fake_df), 40000)
    n_real = min(len(real_df), 39000)
    
    if n_fake > 0 and n_real > 0:
        df = pd.concat([
            fake_df.sample(n=n_fake, random_state=42),
            real_df.sample(n=n_real, random_state=42)
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
    return df