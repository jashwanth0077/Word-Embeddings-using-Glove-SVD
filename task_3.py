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
y_pred = crf.predict(X_test)

f1 = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)
print(f"\nWeighted F1 Score : {f1:.4f}")

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

def analyze_feature_contributions(crf):

    state_features = crf.state_features_
    
    global_importance = defaultdict(float)
    label_importance = defaultdict(lambda: defaultdict(float))
    
    # 1. DEFINE KNOWN FEATURES
    feature_descriptions = {
        'word.lower()': 'Current word',
        'word[-3:]': 'Suffix (last 3 chars)',
        'word[-2:]': 'Suffix (last 2 chars)',
        'word[:3]': 'Prefix (first 3 chars)',
        'word[:2]': 'Prefix (first 2 chars)',
        'word.isupper()': 'Is All Caps?',
        'word.istitle()': 'Is Title Case?',
        'word.isdigit()': 'Is Number?',
        'word.has_hyphen': 'Has Hyphen?',
        'word.length': 'Word Length',
        'postag': 'Current POS tag',
        'postag.is_proper': 'Is Proper Noun?',
        'chunktag': 'Current Chunk tag',
        
        '-1:word.lower()': 'Previous word',
        '-1:word.istitle()': 'Prev word Title Case?',
        '-1:postag': 'Previous POS tag',
        '-1:chunktag': 'Previous Chunk tag',
        
        '+1:word.lower()': 'Next word',
        '+1:word.istitle()': 'Next word Title Case?',
        '+1:postag': 'Next POS tag',
        '+1:chunktag': 'Next Chunk tag',
        
        'bias': 'Bias Term',
        'BOS': 'Start of Sentence',
        'EOS': 'End of Sentence'
    }

    sorted_known_keys = sorted(feature_descriptions.keys(), key=len, reverse=True)

    for (raw_feature, label), weight in state_features.items():
        w = abs(weight)
        
        found_key = "Other"
        
        for key in sorted_known_keys:
            # Check exact match or prefix match
            if raw_feature == key or raw_feature.startswith(key + ":"):
                found_key = key
                break
        
        # Fallback: if it didn't match our list, try splitting by colon
        if found_key == "Other":
             found_key = raw_feature.split(':')[0]

        global_importance[found_key] += w
        label_importance[label][found_key] += w

    ranked_global = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'='*80}")
    print("GLOBAL FEATURE IMPORTANCE (All Tags Combined)")
    print(f"{'='*80}")
    print(f"{'Rank':<5} | {'Feature Type':<25} | {'Description':<25} | {'Total Weight'}")
    print("-" * 80)
    
    for i, (f_type, total_w) in enumerate(ranked_global, 1):
        desc = feature_descriptions.get(f_type, "Custom/Other")
        print(f"{i:<5} | {f_type:<25} | {desc:<25} | {total_w:,.2f}")

    print(f"\n{'='*80}")
    print("TOP 5 FEATURE TYPES PER LABEL")
    print(f"{'='*80}")

    sorted_labels = sorted(label_importance.keys())

    for label in sorted_labels:
        print(f"\nLabel: [{label}]")
        print(f"{'Rank':<5} | {'Feature Type':<25} | {'Importance':<15} | {'% of Label Total'}")
        print("-" * 70)
        
        label_feats = label_importance[label]
        total_label_weight = sum(label_feats.values())
        
        sorted_feats = sorted(label_feats.items(), key=lambda x: x[1], reverse=True)
        
        for i, (f_type, w) in enumerate(sorted_feats[:5], 1):
            percent = (w / total_label_weight) * 100
            print(f"{i:<5} | {f_type:<25} | {w:<15,.2f} | {percent:.1f}%")

analyze_feature_contributions(crf)