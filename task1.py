import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from datasets import load_dataset

# 1. Load CC-News Dataset
ds = load_dataset("sentence-transformers/ccnews", split="train")
column_name = "text" if "text" in ds.column_names else ds.column_names[0]
corpus = ds[column_name][:10000] # Using subset for manageable training

# 2. Build Vocabulary and Co-occurrence Matrix
# Requirement: Case-sensitive (While != while) 
print("Building vocabulary and co-occurrence matrix...")
vocab = sorted(list(set(" ".join(corpus).split())))
vocab_to_idx = {word: i for i, word in enumerate(vocab)}
V_size = len(vocab)

window_size = 5 # Context Window (w) [cite: 14]
cooc_dict = defaultdict(float)

for text in corpus:
    tokens = text.split()
    for i, token in enumerate(tokens):
        idx_i = vocab_to_idx[token]
        
        # Look at window surrounding the word [cite: 14]
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        for j in range(start, end):
            if i == j: continue
            idx_j = vocab_to_idx[tokens[j]]
            # Weighting by distance is a common GloVe convention (1/d)
            cooc_dict[(idx_i, idx_j)] += 1.0 / abs(i - j)

# 3. Prepare Tensors for PyTorch
indices = list(cooc_dict.keys())
i_indices = torch.LongTensor([idx[0] for idx in indices])
j_indices = torch.LongTensor([idx[1] for idx in indices])
x_ij = torch.FloatTensor([cooc_dict[idx] for idx in indices])
log_x_ij = torch.log(x_ij)

# Weighting function f(Xij) [cite: 11, 12]
def get_weights(x, x_max=100, alpha=0.75):
    weights = (x / x_max).pow(alpha)
    return torch.where(x < x_max, weights, torch.ones_like(x))

f_x_ij = get_weights(x_ij)

# 4. GloVe Model Definition [cite: 11]
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.w = nn.Embedding(vocab_size, d)
        self.w_tilde = nn.Embedding(vocab_size, d)
        self.b = nn.Embedding(vocab_size, 1)
        self.b_tilde = nn.Embedding(vocab_size, 1)
        
        # Initialize as per GloVe paper suggestions
        nn.init.uniform_(self.w.weight, -0.5, 0.5)
        nn.init.uniform_(self.w_tilde.weight, -0.5, 0.5)
        nn.init.constant_(self.b.weight, 0)
        nn.init.constant_(self.b_tilde.weight, 0)

    def forward(self, i, j):
        dot_product = torch.sum(self.w(i) * self.w_tilde(j), dim=1)
        return dot_product + self.b(i).squeeze() + self.b_tilde(j).squeeze()

# 5. Training Loop
d = 200 # Fixed dimension for hyperparameter tuning [cite: 18]
model = GloVeModel(V_size, d)
optimizer = optim.Adagrad(model.parameters(), lr=0.05) # Standard for GloVe

print(f"Starting training for d={d}...")
for epoch in range(50):
    optimizer.zero_grad()
    predictions = model(i_indices, j_indices)
    
    # Loss: f(Xij)(wi.T * wj + bi + bj - log Xij)^2 [cite: 11]
    loss = torch.sum(f_x_ij * (predictions - log_x_ij)**2)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# 6. Extract Final Embeddings [cite: 19]
# The GloVe paper suggests w + w_tilde as the final representation
final_embeddings = model.w.weight.detach().numpy() + model.w_tilde.weight.detach().numpy()
print(f"Final Word Representation Matrix Shape: {final_embeddings.shape}")