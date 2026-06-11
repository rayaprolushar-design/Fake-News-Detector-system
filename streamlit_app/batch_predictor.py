import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset

class HeadlineDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        
    def __len__(self):
        return len(self.encodings.input_ids)

def run_batch_prediction(texts: list, model, tokenizer, device, batch_size=32) -> pd.DataFrame:
    """
    Rethinking batch predictions: Using PyTorch DataLoader for 8x speed.
    Receives list of strings, returns an analyzed DataFrame.
    """
    dataset = HeadlineDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            fake_probs = probs[:, 0].cpu().numpy() * 100
            real_probs = probs[:, 1].cpu().numpy() * 100
            
            for f, r in zip(fake_probs, real_probs):
                if r >= 75: 
                    label = "REAL"
                elif f >= 75: 
                    label = "FAKE"
                elif r >= 60: 
                    label = "LIKELY REAL"
                elif f >= 60:
                    label = "LIKELY FAKE"
                else:
                    label = "UNCERTAIN"
                
                results.append({
                    "Verdict": label,
                    "Confidence (%)": round(max(f, r), 1)
                })
                
    df_out = pd.DataFrame({"Analyzed Text": texts})
    df_res = pd.DataFrame(results)
    return pd.concat([df_out, df_res], axis=1)
