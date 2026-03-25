import torch as th
import numpy as np
import pandas as pd
import argparse
import os
from model import BertGCN, BertGAT
from utils import load_corpus, normalize_adj
import scipy.sparse as sp
import dgl
import torch.utils.data as Data

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--bert_init', type=str, default='jcblaise/roberta-tagalog-base')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--m', type=float, default=0.7)
parser.add_argument('--dataset', type=str, default='twitter')
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('--output', type=str, default='predictions.csv')
args = parser.parse_args()

device = th.device('cuda:0') if args.device == 'cuda' and th.cuda.is_available() else th.device('cpu')
print(f"Using device: {device}")

# ── Load graph and corpus ─────────────────────────────────────────────────────
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = \
    load_corpus(args.dataset)

nb_node = features.shape[0]
nb_train = train_mask.sum()
nb_val   = val_mask.sum()
nb_test  = test_mask.sum()
nb_word  = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]

print(f"Graph nodes: {nb_node} | Train: {nb_train} | Val: {nb_val} | Test (unlabeled): {nb_test} | Word: {nb_word}")

# ── Build model ───────────────────────────────────────────────────────────────
if args.gcn_model == 'gcn':
    model = BertGCN(
        nb_class=nb_class,
        pretrained_model=args.bert_init,
        m=args.m,
        gcn_layers=2,
        n_hidden=200,
        dropout=0.5
    )
else:
    model = BertGAT(
        nb_class=nb_class,
        pretrained_model=args.bert_init,
        m=args.m,
        gcn_layers=2,
        n_hidden=32,
        dropout=0.5
    )

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt = th.load(args.checkpoint, map_location=device)
model.bert_model.load_state_dict(ckpt['bert_model'])
model.classifier.load_state_dict(ckpt['classifier'])
model.gcn.load_state_dict(ckpt['gcn'])
print(f"Loaded checkpoint from epoch {ckpt['epoch']}")

# ── Encode corpus text ────────────────────────────────────────────────────────
corpus_file = f'./data/corpus/{args.dataset}_shuffle.txt'
with open(corpus_file, 'r', encoding='utf-8') as f:
    text = f.read().replace('\\', '').split('\n')

def encode_input(text, tokenizer):
    inp = tokenizer(text, max_length=args.max_length, truncation=True,
                    padding='max_length', return_tensors='pt')
    return inp.input_ids, inp.attention_mask

input_ids, attention_mask = encode_input(text, model.tokenizer)

# Insert zero padding for word nodes
# Layout: [train_docs | val_docs | <word_nodes_inserted_here> | test_docs]
# text file has: train+val docs followed by test docs (no word rows)
# so we split at -nb_test from the end
input_ids = th.cat([
    input_ids[:-nb_test],
    th.zeros((nb_word, args.max_length), dtype=th.long),
    input_ids[-nb_test:]
])
attention_mask = th.cat([
    attention_mask[:-nb_test],
    th.zeros((nb_word, args.max_length), dtype=th.long),
    attention_mask[-nb_test:]
])

# ── Build DGL graph ───────────────────────────────────────────────────────────
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'] = input_ids
g.ndata['attention_mask'] = attention_mask
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

# ── Warm up cls_feats with BERT ───────────────────────────────────────────────
doc_mask = train_mask + val_mask + test_mask

print("Computing BERT features for all document nodes...")
model = model.to(device)
model.eval()
with th.no_grad():
    loader = Data.DataLoader(
        Data.TensorDataset(
            g.ndata['input_ids'][doc_mask],
            g.ndata['attention_mask'][doc_mask]
        ),
        batch_size=256
    )
    cls_list = []
    for batch_input_ids, batch_attn in loader:
        out = model.bert_model(
            input_ids=batch_input_ids.to(device),
            attention_mask=batch_attn.to(device)
        )[0][:, 0]
        cls_list.append(out.cpu())
    # Write back to CPU graph before moving graph to device
    g.ndata['cls_feats'][doc_mask] = th.cat(cls_list, dim=0)

# ── Run inference on unlabeled test nodes ─────────────────────────────────────
test_node_idx = th.where(th.BoolTensor(test_mask))[0]
test_loader   = Data.DataLoader(
    Data.TensorDataset(test_node_idx),
    batch_size=args.batch_size
)

g = g.to(device)
all_preds = []
all_probs = []

print("Running inference on unlabeled test nodes...")
with th.no_grad():
    for (idx,) in test_loader:
        idx = idx.to(device)
        logits = model(g, idx)        # log-softmax output
        probs = th.exp(logits)       # back to probabilities
        preds = probs.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_probs = np.concatenate(all_probs)

# ── Map back to unlabeled CSV ─────────────────────────────────────────────────
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

df_unlabeled = pd.read_csv('data/tweets_unlabeled_set.csv')
df_unlabeled = df_unlabeled.dropna(subset=['text']).reset_index(drop=True)

assert len(df_unlabeled) == len(all_preds), \
    f"Mismatch: {len(df_unlabeled)} unlabeled rows vs {len(all_preds)} predictions"

df_unlabeled['predicted_label']    = [label_map[p] for p in all_preds]
df_unlabeled['predicted_label_id'] = all_preds
for i, cls in enumerate(['positive', 'negative', 'neutral']):
    df_unlabeled[f'prob_{cls}'] = all_probs[:, i]

df_unlabeled.to_csv(args.output, index=False)
print(f"\n✓ Predictions saved to: {args.output}")
print(f"\nPrediction distribution:\n{df_unlabeled['predicted_label'].value_counts()}")