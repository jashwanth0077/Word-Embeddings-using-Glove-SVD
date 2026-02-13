import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import svds

# 1. Load the JSON 
# Format assumed: {"word1": [[id1, "text1"], [id2, "text2"]], ...}
json_path = "updated_vocab_document_dict.json"
with open(json_path, "r", encoding="utf-8") as f:
    inverted_data = json.load(f)

# 2. Extract Unique Documents
# We use a dict keyed by doc_id to ensure we don't process the same text twice
unique_docs = {}
for word, doc_list in inverted_data.items():
    for doc_id, doc_text in doc_list:
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc_text

doc_ids = sorted(unique_docs.keys())
print(doc_ids[250:300])  # Print a sample of document IDs
print(f"Total unique documents: {len(unique_docs)}")
corpus = [unique_docs[did] for did in doc_ids]

print(f"Unique documents found: {len(corpus)}")

# 3. Build the Term-Document Matrix A
# We fix the vocabulary to match the keys in your JSON
target_vocab = sorted(inverted_data.keys())
vectorizer = CountVectorizer(vocabulary=target_vocab, lowercase=False)
X_doc_term = vectorizer.fit_transform(corpus)

# Transpose to get A (Size: Vocabulary x Documents)
A = X_doc_term.T.astype(float).tocsc()

# 4. Perform Truncated SVD
d = 200 
U, Sigma, Vt = svds(A,k=d)

# Sort descending
idx = np.argsort(Sigma)[::-1]
U = U[:, idx]
Sigma = Sigma[idx]
X_k = U @ np.diag(Sigma)
word_to_vec = {word: X_k[i].tolist() for i, word in enumerate(target_vocab)}
print(word_to_vec)
print(f"Final Matrix X_k Shape: {X_k.shape}")
