# indic_model.py
# Fine-tunes IndicBERT on Hindi + Telugu fake news data

import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

MODEL_NAME = 'ai4bharat/indic-bert'   # free on Hugging Face

# Support CUDA, MPS (Apple Silicon), and CPU
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(f"Using device: {device}")
print("Downloading IndicBERT tokenizer (~500MB, cached after first run)...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"Error downloading tokenizer: {e}")
    tokenizer = None

class IndicDataset(Dataset):
    def __init__(self, texts, labels, max_len=200):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_indic_model(csv_path: str, save_dir: str = 'indic_model/'):
    """
    Fine-tune IndicBERT on your Hindi+Telugu dataset.
    CSV must have columns: 'text', 'label' (0=fake, 1=real)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}. Run build_hindi_dataset.py first.")
        
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} Hindi/Telugu articles")
    print(df['label'].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'],
        test_size=0.15, stratify=df['label'], random_state=42
    )

    train_ds = IndicDataset(X_train, y_train)
    test_ds  = IndicDataset(X_test,  y_test)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    test_dl  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(device)

    EPOCHS = 3
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps  = len(train_dl) * EPOCHS
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, total_steps // 10, total_steps
    )

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_dl:
            batch    = {k: v.to(device) for k, v in batch.items()}
            outputs  = model(**batch)
            loss     = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds = []
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                preds = model(**batch).logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)

        acc = accuracy_score(y_test.tolist(), all_preds)
        print(f"Epoch {epoch+1} | loss: {total_loss/len(train_dl):.4f} | acc: {acc*100:.2f}%")

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"  ✓ Saved best model to {save_dir}")

    print(f"\nTraining complete! Best accuracy: {best_acc*100:.2f}%")
    return save_dir

if __name__ == '__main__':
    # Local execution guard
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'hindi_telugu_dataset.csv'
    if os.path.exists(csv_file):
        train_indic_model(csv_file)
    else:
        print(f"To run training locally, compile dataset first or pass path: python indic_model.py path/to/dataset.csv")
