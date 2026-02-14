import os
import numpy 
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import save_npz, load_npz
import time
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Global Co-occurrence Matrix Construction ---
def build_global_cooccurrence(json_path, window_size=5):
    """
    Constructs the N x N co-occurrence matrix X_ij.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    vocab = list(data.keys())
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    v_size = len(vocab)
    cooc_dict = {}

    print(f"Building global co-occurrence matrix for {v_size} tokens...")
    
    processed_passages = set()

    for target_word, instances in data.items():
        for instance in instances:
            passage_id = instance[0]
            passage_text = instance[1]
            if passage_id in processed_passages:
                continue
            processed_passages.add(passage_id)

            tokens = passage_text.split()
            
            for i, token in enumerate(tokens):
                if token not in word_to_idx:
                    continue
                
                u = word_to_idx[token]
                
                start = max(0, i - window_size)
                end = min(len(tokens), i + window_size + 1)
                
                for j in range(start, end):
                    if i == j: continue
                    context_word = tokens[j]
                    
                    if context_word in word_to_idx:
                        v = word_to_idx[context_word]
                        weight = 1.0 / abs(i - j)
                        cooc_dict[(u, v)] = cooc_dict.get((u, v), 0) + weight

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
        self.wi = nn.Embedding(vocab_size, d)
        self.wj = nn.Embedding(vocab_size, d)
        self.bi = nn.Embedding(vocab_size, 1)
        self.bj = nn.Embedding(vocab_size, 1)
        
        # Initialization
        for param in self.parameters():
            nn.init.uniform_(param, -0.5, 0.5)

    def forward(self, i, j):
        # Objective: wi^T * wj + bi + bj
        dot_product = torch.sum(self.wi(i) * self.wj(j), dim=1)
        return dot_product + self.bi(i).squeeze() + self.bj(j).squeeze()

def train_glove(matrix, v_size, d=200, lr=0.05, epochs=50, 
                x_max=100, alpha=0.75, batch_size=8192):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GloVeModel(v_size, d).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=lr)

    i_all = matrix.row
    j_all = matrix.col
    x_all = matrix.data
    
    # Pre-calculate log and weights to save GPU cycles
    log_x_all = np.log(x_all)
    weights_all = np.power(x_all / x_max, alpha)
    weights_all[x_all > x_max] = 1.0

    n = len(x_all)
    print(n)
    loss_history = []

    print(f"Training GloVe (d={d}, lr={lr}) on {device}...")

    for epoch in range(epochs):
        perm = np.random.permutation(n)
        epoch_loss = 0

        for start in range(0, n, batch_size):
            indices = perm[start:min(start + batch_size, n)]

            i_batch = torch.LongTensor(i_all[indices]).to(device)
            j_batch = torch.LongTensor(j_all[indices]).to(device)
            x_batch_log = torch.FloatTensor(log_x_all[indices]).to(device)
            w_batch = torch.FloatTensor(weights_all[indices]).to(device)

            optimizer.zero_grad()
            preds = model(i_batch, j_batch)

            diff = preds - x_batch_log
            loss = torch.mean(w_batch * (diff ** 2))/(n / batch_size)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        loss_history.append(epoch_loss)
        print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f}")

    embeddings = (model.wi.weight.detach() + 
                  model.wj.weight.detach()).cpu().numpy()

    return embeddings, loss_history


# --- 4. Main Execution ---
if __name__ == "__main__":

    JSON_FILE = "updated_vocab_document_dict.json" 
    MATRIX_FILE = "cooccurrence_matrix.npz"
    
    if os.path.exists(MATRIX_FILE):
        X_mat = load_npz(MATRIX_FILE).tocoo()
        V_SIZE = X_mat.shape[0]
        print("Reloading vocabulary mapping...")
        with open(JSON_FILE, 'r') as f:
            vocab_list = list(json.load(f).keys())
    else:
        print("Building Global Co-occurrence...")
        X_mat, vocab_list = build_global_cooccurrence(JSON_FILE)
        save_npz("cooccurrence_matrix.npz", X_mat)
        V_SIZE = len(vocab_list)

    print(f"Data Ready. Vocabulary Size: {V_SIZE}")

    # --- REPORT PART 1: Final Hyperparameters ---
    FINAL_LR = 0.05
    FINAL_EPOCHS = 10
    CONTEXT_WINDOW = 5
    ALPHA = 0.75
    X_MAX = 100
    print("\n" + "="*40)
    print("FINAL HYPERPARAMETERS (Tested on Fixed d)")
    print("="*40)
    print(f"Context Window: {CONTEXT_WINDOW}")
    print(f"Alpha:          {ALPHA}")
    print(f"X_max:          {X_MAX}")
    print(f"Learning Rate:  {FINAL_LR}")
    print(f"Iterations:     {FINAL_EPOCHS}")
    print("="*40 + "\n")

    # --- REPORT PART 2: Training Loop ---
    dimensions = [50, 100, 200, 300]
    
    latencies = []
    final_losses = []
    loss_histories = {} 
    
    embeddings_d200 = None

    print(f"Starting training for dimensions: {dimensions}...")
    all_embeddings = {}

    for d_val in dimensions:
        print(f"\n--> Training d={d_val}...")
        
        # --- CALCULATE LATENCY ---
        start_time = time.time()
        vecs, losses = train_glove(X_mat, V_SIZE, d=d_val, lr=FINAL_LR, epochs=FINAL_EPOCHS)
        end_time = time.time()
        all_embeddings[d_val] = vecs
        
        latency = end_time - start_time
        final_loss = losses[-1]
        
        latencies.append(latency)
        final_losses.append(final_loss)
        loss_histories[d_val] = losses
        
        if d_val == 200:
            embeddings_d200 = vecs

        # --- SAVE JSON (Mapped) ---
        filename = f"glove_{d_val}.json"
        embedding_dict = {
            word: vec.tolist() 
            for word, vec in zip(vocab_list, vecs)
        }
        with open(filename, "w") as f:
            json.dump(embedding_dict, f)

        # --- PLOT 1: Individual Loss Curve ---
        plt.figure(figsize=(6, 4))
        plt.plot(losses, color='blue')
        plt.title(f"Loss Curve (d={d_val})")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig(f"plot_loss_d{d_val}.png")
        plt.close()

    # --- REPORT PART 3: Analysis Visualizations ---
    
    plt.figure(figsize=(10, 6))
    for d_val in dimensions:
        plt.plot(loss_histories[d_val], label=f'd={d_val}')
    plt.title("Combined Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plot_combined_loss.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar([str(d) for d in dimensions], latencies, color='orange', alpha=0.7)
    plt.title("Training Latency (Time) vs Dimension")
    plt.xlabel("Dimension (d)")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y')
    plt.savefig("plot_latency_analysis.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(dimensions, final_losses, marker='o', linestyle='-', color='green')
    plt.title("Final Loss vs Dimension (Performance)")
    plt.xlabel("Dimension (d)")
    plt.ylabel("Final Loss")
    plt.grid(True)
    plt.savefig("plot_loss_analysis.png")
    plt.close()

    print("\n[Visuals] All plots saved to disk.")

    print("\n" + "="*40)
    print("LATENCY AND LOSS REPORT")
    print("="*40)
    print(f"{'Dim':<5} | {'Latency (s)':<15} | {'Final Loss':<15}")
    print("-" * 45)
    for i, d in enumerate(dimensions):
        print(f"{d:<5} | {latencies[i]:<15.2f} | {final_losses[i]:<15.4f}")
    print("-" * 45)

 # --- REPORT PART 4: Nearest Neighbors (Top-5 for ALL Dimensions) ---
    print("\n" + "="*60)
    print("NEAREST NEIGHBOR COMPARISON ACROSS DIMENSIONS")
    print("="*60)
    
    word_to_idx = {w: i for i, w in enumerate(vocab_list)}
    idx_to_word = {i: w for i, w in enumerate(vocab_list)}
    
    target_words = ["president", "London", "politics"]
    valid_targets = [w for w in target_words if w in word_to_idx]
    
    if not valid_targets:
        valid_targets = vocab_list[:3]

    sorted_dims = sorted(all_embeddings.keys()) 

    for d_val in sorted_dims:
        print("\n" + "-"*40)
        print(f"Dimension d={d_val}")
        print("-"*40)
        
        current_vecs = all_embeddings[d_val]
        
        for target in valid_targets:
            t_idx = word_to_idx[target]
            
            t_vec = current_vecs[t_idx].reshape(1, -1)
            
            sims = cosine_similarity(t_vec, current_vecs)[0]
            sorted_indices = sims.argsort()[::-1]
            top_indices = sorted_indices[1:6] 
            
            print(f"Word: '{target}'")
            for rank, idx in enumerate(top_indices, 1):
                neighbor_word = idx_to_word[idx]
                score = sims[idx]
                print(f"  {rank}. {neighbor_word:<15} (Sim: {score:.4f})")
            print("")

    print("\nTask Complete.")