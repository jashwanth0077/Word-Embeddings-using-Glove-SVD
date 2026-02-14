import numpy 
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import coo_matrix

# --- 1. Global Co-occurrence Matrix Construction ---
def build_global_cooccurrence(json_path, window_size=5):
    """
    Constructs the N x N co-occurrence matrix X_ij.
    Uses the provided JSON dataset structure.
    """
    with open(json_path, 'r') as f:
        data = json.load(f) # cite: 49, 50
    
    vocab = list(data.keys()) # cite: 51, 53
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    v_size = len(vocab)
    cooc_dict = {}

    print(f"Building global co-occurrence matrix for {v_size} tokens...")
    
    # Track processed passages to avoid double-counting [cite: 52]
    processed_passages = set()

    for target_word, instances in data.items():
        for instance in instances:
            # instance format: [passage_index, passage_text]
            passage_id = instance[0]
            passage_text = instance[1]
            
            if passage_id in processed_passages:
                continue
            processed_passages.add(passage_id)

            # Case-sensitive tokenization 
            tokens = passage_text.split()
            
            for i, token in enumerate(tokens):
                if token not in word_to_idx:
                    continue
                
                u = word_to_idx[token]
                
                # Context Window (w) [cite: 14]
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i == j: continue
                    context_word = tokens[j]
                    
                    if context_word in word_to_idx:
                        v = word_to_idx[context_word]
                        # Weighting by distance 1/d is standard GloVe practice
                        weight = 1.0 / abs(i - j)
                        cooc_dict[(u, v)] = cooc_dict.get((u, v), 0) + weight

    # Convert to COO format for PyTorch [cite: 75]
    rows, cols, data_vals = [], [], []
    for (u, v), count in cooc_dict.items():
        rows.append(u)
        cols.append(v)
        data_vals.append(count)
        
    return coo_matrix((data_vals, (rows, cols)), shape=(v_size, v_size)), vocab

# --- 2. GloVe Model Architecture ---
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, d):
        super(GloVeModel, self).__init__()
        # Embedding matrices of size N x d 
        self.wi = nn.Embedding(vocab_size, d)
        self.wj = nn.Embedding(vocab_size, d)
        # Biases [cite: 11]
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)
        
        # Initialization
        for param in self.parameters():
            nn.init.uniform_(param, -0.5, 0.5)

    def forward(self, i, j):
        # Objective: wi^T * wj + bi + bj [cite: 11]
        dot_product = torch.sum(self.wi(i) * self.wj(j), dim=1)
        return dot_product + self.bi(i).squeeze() + self.bj(j).squeeze()

# --- 3. Training Loop ---
def train_glove(matrix, v_size, d=200, lr=0.05, epochs=50, x_max=100, alpha=0.75):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GloVeModel(v_size, d).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr) # cite: 17
    
    # Prepare tensors
    i_indices = torch.LongTensor(matrix.row).to(device)
    j_indices = torch.LongTensor(matrix.col).to(device)
    x_ij = torch.FloatTensor(matrix.data).to(device)
    log_x_ij = torch.log(x_ij)

    # Weighting function f(Xij) [cite: 12]
    weights = torch.pow(x_ij / x_max, alpha)
    weights = torch.where(x_ij < x_max, weights, torch.ones_like(weights))

    loss_history = []
    print(f"Training GloVe (d={d}, lr={lr})...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(i_indices, j_indices)
        
        # Weighted least-squares loss [cite: 10, 11]
        diff = (preds - log_x_ij)
        loss = torch.sum(weights * (diff**2))
        
        loss.backward()
        optimizer.step()
        
        loss_val = loss.item()
        loss_history.append(loss_val)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss_val:.4f}")
            
    # Combine main and context vectors [cite: 19]
    embeddings = (model.wi.weight.data + model.wj.weight.data).cpu().numpy()
    return embeddings, loss_history

# --- 4. Main Execution ---
if __name__ == "__main__":
    JSON_FILE = "updated_vocab_document_dict.json" 
    
    # Build Matrix X
    X_mat, vocab_list = build_global_cooccurrence(JSON_FILE)
    V_SIZE = len(vocab_list)
    
    # 1. First, optimize for d=200 [cite: 18]
    # Adjust lr and epochs here for hyperparameter report [cite: 62]
    best_d200_vecs, _ = train_glove(X_mat, V_SIZE, d=200, lr=0.05, epochs=50)
    
    # 2. Generate required dimensions for Task 4 [cite: 19]
    for d_val in [50, 100, 200, 300]:
        vecs, losses = train_glove(X_mat, V_SIZE, d=d_val)
        np.save(f"glove_vecs_{d_val}.npy", vecs)
        # Note: Save 'losses' to plot curves for the report 
    
    print("Task 1 complete. All embeddings saved as .npy files.")