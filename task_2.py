import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse.linalg import svds
from sklearn.metrics.pairwise import cosine_similarity


# 1. Loading the JSON 
json_path = "updated_vocab_document_dict.json"
with open(json_path, "r", encoding="utf-8") as f:
    inverted_data = json.load(f)

# 2. Extract Unique Documents
unique_docs = {}
for word, doc_list in inverted_data.items():
    for doc_id, doc_text in doc_list:
        if doc_id not in unique_docs:
            unique_docs[doc_id] = doc_text

doc_ids = sorted(unique_docs.keys())  
corpus = []
for did in doc_ids:
    corpus.append(unique_docs[did])

print(f"Unique documents found: {len(corpus)}")

target_vocab = sorted(inverted_data.keys())
vectorizer = CountVectorizer(vocabulary=target_vocab, lowercase=False)
X_doc_term = vectorizer.fit_transform(corpus)

A = X_doc_term.T.astype(float)
print(f"Term-Document Matrix A Shape: {A.shape}")

def get_top_n_neighbors(target_word, vector_matrix, n=5):
    if target_word not in target_vocab:
        return f"Word '{target_word}' not found in vocabulary."
    
    word_to_index = {}
    for i, w in enumerate(target_vocab):
            word_to_index[w] = i
    word_idx = word_to_index[target_word]
    word_vec = vector_matrix[word_idx].reshape(1, -1)
    
    similarities = cosine_similarity(word_vec, vector_matrix).flatten()
    related_indices = similarities.argsort()[::-1][1:n+1]
    neighbors = [(target_vocab[i], similarities[i]) for i in related_indices]
    return neighbors

# 4. Perform Truncated SVD
Dim = [50, 100, 200,300]
for d in Dim:
    U, Sigma, Vt = svds(A,k=d)

    idx = np.argsort(Sigma)[::-1]
    U = U[:, idx]
    Sigma = Sigma[idx]
    X_k = U @ np.diag(Sigma)
    word_to_vec = {}
    for i, word in enumerate(target_vocab):
        word_to_vec[word] = X_k[i].tolist()


    with open(f"svd_{d}.json", "w", encoding="utf-8") as f:
        json.dump(word_to_vec, f, ensure_ascii=False, indent=4)

    vector_matrix = []
    for word in target_vocab:
            vector_matrix.append(word_to_vec[word])
    vector_matrix = np.array(vector_matrix)

    chosen_words = ["president", "London", "politics"] 

    print(f"\n--- Top 5 Nearest Neighbors (SVD {d}) ---")
    for word in chosen_words:
        neighbors = get_top_n_neighbors(word, vector_matrix, n=5)
        print(f"\nWord: {word}")
        if isinstance(neighbors, list):
            for neighbor, score in neighbors:
                print(f"  -> {neighbor}: {score:.4f}")
        else:
            print(neighbors)




