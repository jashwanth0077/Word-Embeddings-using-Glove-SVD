import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import svds

# 1. Load Local Parquet Data
from datasets import load_dataset

ds = load_dataset("sentence-transformers/ccnews", split="train")
# This will download the actual binary parquet files for you automatically
column_name = "title" if "title" in ds.column_names else ds.column_names[0]
print(f"Using column: {column_name}")

# Taking a subset for Task 2 to keep SVD computation manageable
corpus = ds[column_name]
# 2. Build the Term-Document Matrix A
# max_features ensures the vocabulary size V is controlled
V_size = 10000 
vectorizer = CountVectorizer(max_features=V_size, stop_words='english')
X_doc_term = vectorizer.fit_transform(corpus)

# Transpose to get Term-Document Matrix A (Size: V x N)
A = X_doc_term.transpose().astype(float)

# 3. Perform Truncated SVD (Rank d approximation)
d = 200 
U, Sigma, Vt = svds(A, k=d)

# Re-order: svds returns values in ascending order; we want descending
idx = np.argsort(Sigma)[::-1]
U = U[:, idx]
Sigma = Sigma[idx]


X_k = U @ np.diag(Sigma)

# 5. Result Mapping
vocab = vectorizer.get_feature_names_out()
word_to_vec = {word: X_k[i] for i, word in enumerate(vocab)}

print(f"Processed {len(corpus)} documents.")
print(f"Final Word Representation Matrix Shape: {X_k.shape}")