"""
Build graph for Twitter dataset with conversation connections
Modified from original build_graph.py to include tweet-tweet edges based on conversations
"""
import os
import random
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, help='Dataset name')
parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--conversation_weight', type=float, default=1.0, 
                   help='Weight for conversation edges (default: 1.0)')
args = parser.parse_args()

dataset = args.dataset
seed = args.seed
conversation_weight = args.conversation_weight

# Set random seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
print(f"Using random seed: {seed}")
print(f"Conversation edge weight: {conversation_weight}")

# Check if this is a Twitter dataset (has conversation mapping)
conversation_map_path = f'data/{dataset}_conversations.pkl'
has_conversations = os.path.exists(conversation_map_path)

if has_conversations:
    print(f"✓ Found conversation mapping: {conversation_map_path}")
    with open(conversation_map_path, 'rb') as f:
        conversation_map = pkl.load(f)
    print(f"  Loaded {len(conversation_map)} conversations")
else:
    print("⚠ No conversation mapping found, building standard graph")
    conversation_map = {}

word_embeddings_dim = 300
word_vector_map = {}

# Read document names
doc_name_list = []
doc_train_list = []
doc_val_list = []
doc_test_list = []  # This will be unlabeled data

f = open('data/' + dataset + '.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_name_list.append(line.strip())
    temp = line.split("\t")
    if temp[1].find('train') != -1:
        doc_train_list.append(line.strip())
    elif temp[1].find('val') != -1:
        doc_val_list.append(line.strip())
    elif temp[1].find('test') != -1:
        doc_test_list.append(line.strip())
f.close()

print(f"Data splits:")
print(f"  Train (labeled): {len(doc_train_list)}")
print(f"  Val (labeled): {len(doc_val_list)}")
print(f"  Test (unlabeled): {len(doc_test_list)}")

# Read document content
doc_content_list = []
f = open('data/corpus/' + dataset + '.clean.txt', 'r', encoding='utf-8')
lines = f.readlines()
for line in lines:
    doc_content_list.append(line.strip())
f.close()

# Create train indices
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(f"Train samples: {len(train_ids)}")
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.train.index', 'w')
f.write(train_ids_str)
f.close()

# Create val indices
val_ids = []
for val_name in doc_val_list:
    val_id = doc_name_list.index(val_name)
    val_ids.append(val_id)
print(f"Validation samples: {len(val_ids)}")
random.shuffle(val_ids)

val_ids_str = '\n'.join(str(index) for index in val_ids)
f = open('data/' + dataset + '.val.index', 'w')
f.write(val_ids_str)
f.close()

# Create test indices (unlabeled data)
test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(f"Test samples (unlabeled): {len(test_ids)}")
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.test.index', 'w')
f.write(test_ids_str)
f.close()

# Combine all in order: train, val, test
ids = train_ids + val_ids + test_ids
train_size = len(train_ids) + len(val_ids)  # train_size includes validation for BertGCN
test_size = len(test_ids)

print(f"\nTotal samples: {len(ids)}")
print(f"Train+Val size (for graph): {train_size}")
print(f"Test size (unlabeled): {test_size}")

# Shuffle documents
shuffle_doc_name_list = []
shuffle_doc_words_list = []
print("Shuffling documents...")
for id in tqdm(ids, desc="Processing documents"):
    shuffle_doc_name_list.append(doc_name_list[int(id)])
    shuffle_doc_words_list.append(doc_content_list[int(id)])

shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)

f = open('data/' + dataset + '_shuffle.txt', 'w', encoding='utf-8')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '_shuffle.txt', 'w', encoding='utf-8')
f.write(shuffle_doc_words_str)
f.close()

# Build vocabulary
print("Building vocabulary...")
word_freq = {}
word_set = set()
for doc_words in tqdm(shuffle_doc_words_list, desc="Building vocab"):
    words = doc_words.split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

word_doc_list = {}

print("Computing word-document frequencies...")
for i in tqdm(range(len(shuffle_doc_words_list)), desc="Word-doc freq"):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '_vocab.txt', 'w', encoding='utf-8')
f.write(vocab_str)
f.close()

# Label list
label_set = set()
for doc_meta in shuffle_doc_name_list:
    temp = doc_meta.split('\t')
    label = temp[2]
    if label != 'unlabeled':  # Skip unlabeled placeholder
        label_set.add(label)
label_list = list(label_set)

label_list_str = '\n'.join(label_list)
f = open('data/corpus/' + dataset + '_labels.txt', 'w', encoding='utf-8')
f.write(label_list_str)
f.close()

print(f"Labels: {label_list}")

# Training data features (x)
real_train_size = len(train_ids)
val_size = len(val_ids)

real_train_doc_names = shuffle_doc_name_list[:real_train_size]
real_train_doc_names_str = '\n'.join(real_train_doc_names)

f = open('data/' + dataset + '.real_train.name', 'w', encoding='utf-8')
f.write(real_train_doc_names_str)
f.close()

row_x = []
col_x = []
data_x = []
print("Building training features...")
for i in tqdm(range(real_train_size), desc="Train features"):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        data_x.append(doc_vec[j] / doc_len)

x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(f"Training labels shape: {y.shape}")

# Test data features (tx)
row_tx = []
col_tx = []
data_tx = []
print("Building test features...")
for i in tqdm(range(test_size), desc="Test features"):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        data_tx.append(doc_vec[j] / doc_len)

tx = sp.csr_matrix((data_tx, (row_tx, col_tx)), shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    doc_meta = shuffle_doc_name_list[i + train_size]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    # Test set is unlabeled, so all zeros (no label)
    ty.append(one_hot)
ty = np.array(ty)
print(f"Test labels shape: {ty.shape} (all unlabeled, zeros only)")

# All training features (allx)
word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

print("Building all training features...")
for i in tqdm(range(train_size), desc="All train features"):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_len = len(words)
    if doc_len == 0:
        doc_len = 1
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        data_allx.append(doc_vec[j] / doc_len)

for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), 
                     shape=(train_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size):
    doc_meta = shuffle_doc_name_list[i]
    temp = doc_meta.split('\t')
    label = temp[2]
    one_hot = [0 for l in range(len(label_list))]
    # Train and val have real labels
    label_index = label_list.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

# Word nodes get zero labels
for i in range(vocab_size):
    one_hot = [0 for l in range(len(label_list))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

# Build heterogeneous graph with conversation edges
window_size = 20
windows = []

print("Creating sliding windows...")
for doc_words in tqdm(shuffle_doc_words_list, desc="Sliding windows"):
    words = doc_words.split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)

word_window_freq = {}
print("Computing word window frequencies...")
for window in tqdm(windows, desc="Window freq"):
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
print("Computing word pair counts...")
for window in tqdm(windows, desc="Word pairs"):
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

num_window = len(windows)

print("Computing PMI weights for word-word edges...")
for key in tqdm(word_pair_count, desc="PMI weights"):
    temp = key.split(',')
    i = int(temp[0])
    j = int(temp[1])
    count = word_pair_count[key]
    word_freq_i = word_window_freq[vocab[i]]
    word_freq_j = word_window_freq[vocab[j]]
    pmi = log((1.0 * count / num_window) /
              (1.0 * word_freq_i * word_freq_j/(num_window * num_window)))
    if pmi <= 0:
        continue
    row.append(train_size + i)
    col.append(train_size + j)
    weight.append(pmi)

# Doc-word edges
doc_word_freq = {}

print("Computing doc-word frequencies...")
for doc_id in tqdm(range(len(shuffle_doc_words_list)), desc="Doc-word freq"):
    doc_words = shuffle_doc_words_list[doc_id]
    words = doc_words.split()
    for word in words:
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

print("Building doc-word graph edges...")
for i in tqdm(range(len(shuffle_doc_words_list)), desc="Doc-word edges"):
    doc_words = shuffle_doc_words_list[i]
    words = doc_words.split()
    doc_word_set = set()
    for word in words:
        if word in doc_word_set:
            continue
        j = word_id_map[word]
        key = str(i) + ',' + str(j)
        freq = doc_word_freq[key]
        if i < train_size:
            row.append(i)
        else:
            row.append(i + vocab_size)
        col.append(train_size + j)
        idf = log(1.0 * len(shuffle_doc_words_list) / word_doc_freq[vocab[j]])
        weight.append(freq * idf)
        doc_word_set.add(word)

# Add conversation edges (doc-doc connections)
if has_conversations and conversation_map:
    print("\n" + "="*60)
    print("Adding conversation-based edges (NEW!)")
    print("="*60)
    
    # Create mapping from original doc IDs to shuffled positions
    original_to_shuffled = {}
    for shuffled_pos, orig_id in enumerate(ids):
        original_to_shuffled[orig_id] = shuffled_pos
    
    conversation_edges_added = 0
    
    for conv_id, doc_ids in tqdm(conversation_map.items(), desc="Conversation edges"):
        if len(doc_ids) < 2:
            continue
        
        # Get shuffled positions for docs in this conversation
        shuffled_positions = []
        for orig_id in doc_ids:
            if orig_id in original_to_shuffled:
                shuffled_pos = original_to_shuffled[orig_id]
                shuffled_positions.append(shuffled_pos)
        
        # Connect all pairs in the conversation
        for i in range(len(shuffled_positions)):
            for j in range(i + 1, len(shuffled_positions)):
                pos_i = shuffled_positions[i]
                pos_j = shuffled_positions[j]
                
                # Adjust positions based on train/test split
                if pos_i < train_size:
                    node_i = pos_i
                else:
                    node_i = pos_i + vocab_size
                
                if pos_j < train_size:
                    node_j = pos_j
                else:
                    node_j = pos_j + vocab_size
                
                # Add bidirectional edges
                row.append(node_i)
                col.append(node_j)
                weight.append(conversation_weight)
                
                row.append(node_j)
                col.append(node_i)
                weight.append(conversation_weight)
                
                conversation_edges_added += 2
    
    print(f"✓ Added {conversation_edges_added} conversation edges")
    print(f"  Average edges per conversation: {conversation_edges_added / len(conversation_map):.2f}")

node_size = train_size + vocab_size + test_size
print(f"\nTotal nodes in graph: {node_size}")
print(f"  Train+Val docs: {train_size}")
print(f"  Words: {vocab_size}")
print(f"  Test+Unlabeled docs: {test_size}")

print("Creating adjacency matrix...")
adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

print(f"Graph edges: {len(row)}")
print(f"Graph density: {len(row) / (node_size * node_size):.6f}")

# Save all data
print("\nSaving data files...")
f = open("data/ind.{}.x".format(dataset), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.y".format(dataset), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.tx".format(dataset), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.ty".format(dataset), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.allx".format(dataset), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.ally".format(dataset), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.adj".format(dataset), 'wb')
pkl.dump(adj, f)
f.close()

print("\n✓ Graph building complete!")
print(f"  Files saved to: data/ind.{dataset}.*")