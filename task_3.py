import sklearn_crfsuite
from sklearn_crfsuite import metrics
from datasets import load_dataset
from collections import Counter

# 1. FEATURE ENGINEERING (Lexical, Shape, Sub-word, POS, Chunk)
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]   
    chunktag = sent[i][2]  
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        #  SUB-WORD FEATURES 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        # SHAPE FEATURES 
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.has_hyphen': '-' in word,
        'word.length': len(word),
        # GRAMMATICAL FEATURES (POS & CHUNK)
        'postag': postag,
        'postag.is_proper': postag in [22, 23], 
        'chunktag': chunktag,                 
    }
    
    # --- CONTEXT WINDOW (i-1) ---
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        chunktag1 = sent[i-1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:postag': postag1,
            '-1:chunktag': chunktag1,
        })
    else:
        features['BOS'] = True

    # --- CONTEXT WINDOW (i+1) ---
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        chunktag1 = sent[i+1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:postag': postag1,
            '+1:chunktag': chunktag1,
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
  
    return [label for token, postag, chunktag, label in sent]

# 2. DATA PREPARATION

print("Loading CoNLL-2003 dataset...")
dataset = load_dataset("lhoestq/conll2003") 

def prepare_data(data_split):
    label_map = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    formatted_data = []
    for example in data_split:
        tokens = example['tokens']
        pos_tags = example['pos_tags']
        chunk_tags = example['chunk_tags']
        ner_ids = example['ner_tags']
        
      
        labels = [label_map[i] for i in ner_ids]
        
       
        formatted_data.append(list(zip(tokens, pos_tags, chunk_tags, labels)))
    return formatted_data

train_sents = prepare_data(dataset['train'])
test_sents = prepare_data(dataset['test'])
print(f"Processing {len(train_sents)} training sentences...")
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

print(f"Processing {len(test_sents)} test sentences...")
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# 3. TRAINING
print("Training CRF model (this may take 1-2 mins)...")
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,
    c2=0.1,
    max_iterations=100,
    all_possible_transitions=True
)
crf.fit(X_train, y_train)

# 4. ANALYSIS: FEATURE IMPORTANCE
def print_top_features(crf):
    state_features = crf.state_features_
    sorted_features = sorted(state_features.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("\n--- TOP 100 MOST INFLUENTIAL FEATURES ---")
    print(f"{'Weight':>8} | {'Label':<6} | {'Feature Name'}")
    print("-" * 50)
    for (feature, label), weight in sorted_features[:100]:
        print(f"{weight:8.4f} | {label:<6} | {feature}")

print_top_features(crf)

# 5. EVALUATION
labels = list(crf.classes_)
labels.remove('O') 
y_pred = crf.predict(X_test)

f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
print(f"\nWeighted F1 Score (excluding 'O'): {f1:.4f}")

# Detailed report
print("\nClassification Report:")
print(metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3))

# --- SUMMARY FOR ASSIGNMENT ---
print("\n=== SUMMARY OF INCLUDED FEATURES ===")
print("1. LEXICAL:   word.lower(), word neighbors (-1, +1), BOS, EOS")
print("2. SHAPE:     isupper(), istitle(), isdigit(), has_hyphen, length")
print("3. SUB-WORD:  Prefixes ([:2], [:3]), Suffixes ([-2:], [-3:])")
print("4. GRAMMAR:   POS Tags (postag), Proper Noun Flag (postag.is_proper)")
print("5. STRUCTURAL: Chunk Tags (chunktag) - identifying Noun/Verb phrases")

from collections import defaultdict

def print_final_feature_contributions(crf):
    """
    Aggregates and prints the sum of absolute weights for each feature type.
    Fixes AttributeError by unpacking (feature, label) tuples from state_features_.
    """
    state_features = crf.state_features_
    type_importance = defaultdict(float)
    
    # 1. DEFINE KNOWN FEATURES (Exact keys from word2features)
    feature_descriptions = {
        'word.lower()': 'The current word (lowercase)',
        'word[-3:]': 'Suffix: last 3 characters',
        'word[-2:]': 'Suffix: last 2 characters',
        'word[:3]': 'Prefix: first 3 characters',
        'word[:2]': 'Prefix: first 2 characters',
        'word.isupper()': 'Is word all caps?',
        'word.istitle()': 'Is word title-cased?',
        'word.isdigit()': 'Is word a number?',
        'word.has_hyphen': 'Does word contain a hyphen?',
        'word.length': 'Word length',
        'postag': 'Current POS tag',
        'postag.is_proper': 'Is it a proper noun (NNP/NNPS)?',
        'chunktag': 'Current Chunk tag',
        '-1:word.lower()': 'Previous word (lowercase)',
        '-1:word.istitle()': 'Was previous word title-cased?',
        '-1:postag': 'Previous POS tag',
        '-1:chunktag': 'Previous Chunk tag',
        '+1:word.lower()': 'Next word (lowercase)',
        '+1:word.istitle()': 'Is next word title-cased?',
        '+1:postag': 'Next POS tag',
        '+1:chunktag': 'Next Chunk tag',
        'bias': 'Model bias term',
        'BOS': 'Beginning of Sentence marker',
        'EOS': 'End of Sentence marker'
    }

    
    sorted_known_keys = sorted(feature_descriptions.keys(), key=len, reverse=True)

    # 2. UNPACK TUPLE AND AGGREGATE
   
    for (raw_feature, label), weight in state_features.items():
        w = abs(weight)
        match_found = False

       
        for key in sorted_known_keys:
            if raw_feature == key or raw_feature.startswith(key + ":"):
                type_importance[key] += w
                match_found = True
                break
        
        if not match_found:
            
            clean_key = raw_feature.split(':')[0]
            type_importance[clean_key] += w

    # 3. SORT & PRINT
    ranked_features = sorted(type_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<5} | {'Feature Type':<25} | {'What it means':<35} | {'Total Importance'}")
    print("-" * 100)
    
    for i, (f_type, total_w) in enumerate(ranked_features, 1):
        desc = feature_descriptions.get(f_type, "Grammar/Context Value")
        print(f"{i:<5} | {f_type:<25} | {desc:<35} | {total_w:,.2f}")

print_final_feature_contributions(crf)