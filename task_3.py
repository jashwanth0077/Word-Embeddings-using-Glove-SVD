import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset # Standard HF loader

# 1. FEATURE ENGINEERING LOGIC
def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        # --- SUB-WORD FEATURES ---
        'word[-3:]': word[-3:], # Suffix
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],   # Prefix
        'word[:2]': word[:2],
        
        # --- SHAPE FEATURES ---
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        
        # --- INNOVATION: MAPPING SHAPES & SPECIAL CHARS ---
        # These help detect ORG (e.g., "IBM-UK") or LOC/MISC
        'word.has_hyphen': '-' in word,
        'word.is_punctuation': all(c in '.,!?;:"' for c in word),
        'word.length': len(word),
    }
    
    # --- CONTEXT WINDOW (Lexical & Shape) ---
    # Looking at the previous word
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True # Beginning of Sentence

    # Looking at the next word
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True # End of Sentence

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

# 2. DATA LOADING
# Note: Using your local conll2003.py or the standard HF loader
print("Loading CoNLL-2003 dataset...")
dataset = load_dataset("lhoestq/conll2003")

def prepare_data(data_split):
    # 1. Manually define the CoNLL-2003 labels since they aren't in your metadata
    # The order here is CRITICAL and follows the standard CoNLL-2003 mapping
    label_map = [
        "O",       # 0
        "B-PER",   # 1
        "I-PER",   # 2
        "B-ORG",   # 3
        "I-ORG",   # 4
        "B-LOC",   # 5
        "I-LOC",   # 6
        "B-MISC",  # 7
        "I-MISC"   # 8
    ]
    
    formatted_data = []
    
    # 2. Iterate through the examples
    for example in data_split:
        tokens = example['tokens']
        ner_ids = example['ner_tags']
        
        # Convert integer IDs to the string labels defined above
        # Using a list comprehension for efficiency
        labels = [label_map[i] for i in ner_ids]
        
        # 3. Zip them together into (token, label) pairs
        formatted_data.append(list(zip(tokens, labels)))
        
    return formatted_data

# Example usage:
# train_sents = prepare_data(dataset['train'])

train_sents = prepare_data(dataset['train'])
test_sents = prepare_data(dataset['test'])

X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# 3. MODEL TRAINING
print("Training CRF model (this may take a minute)...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1, # L1 regularization
    c2=0.1, # L2 regularization
    max_iterations=100,
    all_possible_transitions=True # Innovation: Helps with illegal transitions like O -> I-PER
)
crf.fit(X_train, y_train)

# 4. EVALUATION
labels = list(crf.classes_)
labels.remove('O') # We usually care more about the entity F1 than the 'O' tag

y_pred = crf.predict(X_test)
f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

print(f"\nWeighted F1 Score (excluding 'O'): {f1:.4f}")

# Detailed report
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))