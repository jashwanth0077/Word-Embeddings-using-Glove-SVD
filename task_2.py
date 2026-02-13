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

doc_ids = sorted(unique_docs.keys())  # Print a sample of document IDs

corpus = [unique_docs[did] for did in doc_ids]

print(f"Unique documents found: {len(corpus)}")

# 3. Build the Term-Document Matrix A
# We fix the vocabulary to match the keys in your JSON
target_vocab = sorted(inverted_data.keys())
vectorizer = CountVectorizer(vocabulary=target_vocab, lowercase=False)
X_doc_term = vectorizer.fit_transform(corpus)

# Transpose to get A (Size: Vocabulary x Documents)
A = X_doc_term.T.astype(float)
print(f"Term-Document Matrix A Shape: {A.shape}")
# 4. Perform Truncated SVD
d = 200 
U, Sigma, Vt = svds(A,k=d)

# Sort descending
idx = np.argsort(Sigma)[::-1]
U = U[:, idx]
Sigma = Sigma[idx]
X_k = U @ np.diag(Sigma)
word_to_vec = {word: X_k[i].tolist() for i, word in enumerate(target_vocab)}
print(word_to_vec["0"])
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ... (Previous code remains the same until word_to_vec)

# 1. Convert our word_to_vec dictionary back into a matrix for fast computation
# We use target_vocab to ensure the order is correct
vector_matrix = np.array([word_to_vec[w] for w in target_vocab])

def get_top_n_neighbors(target_word, n=5):
    if target_word not in target_vocab:
        return f"Word '{target_word}' not found in vocabulary."
    
    # Get the index and vector for the target word
    word_idx = target_vocab.index(target_word)
    word_vec = vector_matrix[word_idx].reshape(1, -1)
    
    # Calculate cosine similarity between the target word and ALL words in the vocab
    # Result is an array of similarity scores
    similarities = cosine_similarity(word_vec, vector_matrix).flatten()
    
    # Sort indices by similarity score in descending order
    # [1:] because the most similar word is always the word itself (similarity = 1.0)
    related_indices = similarities.argsort()[::-1][1:n+1]
    
    neighbors = [(target_vocab[i], similarities[i]) for i in related_indices]
    return neighbors

# 2. Chosen Words (Replace these with the 3 words you used in Task 1)
chosen_words = ["market", "news", "politics"] 

print("\n--- Top 5 Nearest Neighbors (SVD) ---")
for word in chosen_words:
    neighbors = get_top_n_neighbors(word)
    print(f"\nWord: {word}")
    if isinstance(neighbors, list):
        for neighbor, score in neighbors:
            print(f"  -> {neighbor}: {score:.4f}")
    else:
        print(neighbors)

# print(word_to_vec)
print(f"Final Matrix X_k Shape: {X_k.shape}")
