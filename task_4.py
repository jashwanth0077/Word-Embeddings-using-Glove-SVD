import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report, accuracy_score

dataset = load_dataset("lhoestq/conll2003")
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

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

def get_xy_conll_advanced(dataset_split, embeddings, dim):
    X, y = [], []
    
    all_vectors = np.array(list(embeddings.values()))
    mean_vector = np.mean(all_vectors, axis=0)
    lower_embeddings = {}
    for k, v in embeddings.items():
        lower_key = k.lower()
        lower_embeddings[lower_key] = v

    oov_count = 0
    total_count = 0
    for example in dataset_split:
        for word, tag_id in zip(example['tokens'], example['ner_tags']):
            total_count += 1
            if word in embeddings:
                vec = embeddings[word]
            elif word.lower() in lower_embeddings:
                vec = lower_embeddings[word.lower()]
            else:
                vec = mean_vector
                oov_count += 1
            X.append(vec)
            y.append(tag_id)
            
    print(f"OOV Rate: {oov_count/total_count:.2%}")
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(y))

def run_experiment(X_train, y_train, X_test, y_test, dim):
    model = NER_MLP(dim, len(TAGS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    for epoch in range(10):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test)
        preds = torch.argmax(test_logits, dim=1).cpu().numpy()
        y_true = y_test.cpu().numpy()

    acc = accuracy_score(y_true, preds)
    report = classification_report(y_true, preds, target_names=TAGS, digits=4)
    
    return acc, report

DIMENSIONS = [50, 100, 200, 300]
ALGORITHMS = ["GloVE","SVD"]
for dim in DIMENSIONS:
    for algo in ALGORITHMS:
        file_path = f"{algo.lower()}_{dim}.json"
        try:
            with open(file_path, "r") as f:
                embeddings = json.load(f)
            
            print(f"\n--- Running {algo} at {dim}d ---")
            X_train, y_train = get_xy_conll_advanced(dataset['train'], embeddings, dim)
            X_test, y_test = get_xy_conll_advanced(dataset['test'], embeddings, dim)
            
            accuracy, full_report = run_experiment(X_train, y_train, X_test, y_test, dim)
            
            print(f"Overall Accuracy: {accuracy:.4f}")
            print("Detailed Classification Report:")
            print(full_report)
            
        except FileNotFoundError:
            print(f"Skipping {file_path}") 