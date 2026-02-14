import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import pandas as pd
from datasets import load_dataset

# 1. LOAD CoNLL 2003 DATASET
print("Loading CoNLL 2003 dataset...")



dataset = load_dataset("lhoestq/conll2003")


TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
tag_to_idx = {tag: i for i, tag in enumerate(TAGS)}

# 2. MLP MODEL DEFINITION
class NER_MLP(nn.Module):
    def __init__(self, input_dim, num_tags):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1), 
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_tags)
        )
        
    def forward(self, x):
        return self.network(x)

# 3. DATA LOADING HELPER
# Hugging Face CoNLL format provides ['tokens'] and ['ner_tags'] (already as IDs)
def get_xy_conll(dataset_split, embeddings, dim):
    X, y = [], []
    for example in dataset_split:
        tokens = example['tokens']
        ner_tags = example['ner_tags']
        
        for word, tag_id in zip(tokens, ner_tags):
            # Check for word in embeddings, fallback to zero vector
            vec = np.array(embeddings.get(word, np.zeros(dim)))
            X.append(vec)
            y.append(tag_id)
            
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

# 4. TRAINING FUNCTION
def run_training(X_train, y_train, X_test, y_test, dim):
    model = NER_MLP(dim, len(TAGS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    
    for epoch in range(10): # Reduced epochs for faster execution on large CoNLL data
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy

# 5. EXPERIMENT LOOP
DIMENSIONS = [50, 100, 200, 300]
ALGORITHMS = ["SVD"]
results = []

for dim in DIMENSIONS:
    for algo in ALGORITHMS:
        file_path = f"{algo.lower()}_{dim}.json"
        try:
            with open(file_path, "r") as f:
                embeddings = json.load(f)
            
            print(f"Processing {algo} {dim}d...")
            X_train, y_train = get_xy_conll(dataset['train'], embeddings, dim)
            X_test, y_test = get_xy_conll(dataset['test'], embeddings, dim)
            
            acc = run_training(X_train, y_train, X_test, y_test, dim)
            results.append({"Algorithm": algo, "Dimension": dim, "Accuracy": f"{acc:.4f}"})
            
        except FileNotFoundError:
            print(f"Warning: {file_path} not found.")

# 6. RESULTS
df_results = pd.DataFrame(results)
print("\nFinal Comparison Table (CoNLL 2003):")
print(df_results.pivot(index="Dimension", columns="Algorithm", values="Accuracy"))