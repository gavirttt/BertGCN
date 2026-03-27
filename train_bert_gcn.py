import torch as th
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F
from utils import *
import dgl
import torch.utils.data as Data
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer, Engine
from ignite.metrics import Accuracy, Loss, F1
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import numpy as np
import os
import shutil
import argparse
import sys
import logging
import json
from datetime import datetime
from torch.optim import lr_scheduler
from model import BertGCN, BertGAT
from tqdm import tqdm
from ignite.contrib.handlers import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('--max_length', type=int, default=128, help='the input length for bert')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('-m', '--m', type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
parser.add_argument('--nb_epochs', type=int, default=50)
parser.add_argument('--bert_init', type=str, default='roberta-base',
                    choices=['roberta-base', 'roberta-large', 'bert-base-uncased', 'bert-large-uncased', 'jcblaise/roberta-tagalog-base'])
parser.add_argument('--pretrained_bert_ckpt', default=None)
parser.add_argument('--dataset', default='20ng',
                    help='Dataset name. Accepts any string — standard names '
                         '(20ng, R8, R52, ohsumed, mr, isarcasm, semeval3a, twitter) '
                         'or fold-specific names for k-fold CV.')
parser.add_argument('--checkpoint_dir', default=None, help='checkpoint directory, [bert_init]_[gcn_model]_[dataset] if not specified')
parser.add_argument('--gcn_model', type=str, default='gcn', choices=['gcn', 'gat'])
parser.add_argument('--gcn_layers', type=int, default=2)
parser.add_argument('--n_hidden', type=int, default=200, help='the dimension of gcn hidden layer, the dimension for gat is n_hidden * heads')
parser.add_argument('--heads', type=int, default=8, help='the number of attentionn heads for gat')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gcn_lr', type=float, default=1e-3)
parser.add_argument('--bert_lr', type=float, default=1e-5)
parser.add_argument('--seed', type=int, default=42, help='random seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device to use for training')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--use_custom_test', action='store_true', help='use custom test set for semeval3a dataset')
parser.add_argument('--encoding', type=str, default='utf-8', help='file encoding for corpus files')

# ── K-fold: override train/test masks at runtime ────────────────────
# When these are supplied, load_corpus() still loads the full graph (ind.* files
# built from ALL documents). Only the masks are replaced with the fold indices.
# The graph structure — nodes, edges, word nodes — remains unchanged.
parser.add_argument('--train_indices', type=str, default=None,
                    help='Path to text file with train doc indices (one per line). '
                         'When provided, overrides the train mask from load_corpus().')
parser.add_argument('--test_indices', type=str, default=None,
                    help='Path to text file with test doc indices (one per line). '
                         'When provided, overrides the test mask from load_corpus().')
parser.add_argument('--current_fold', type=int, default=0, help='Current fold number for k-fold CV')
parser.add_argument('--total_folds', type=int, default=0, help='Total number of folds for k-fold CV')

args = parser.parse_args()
max_length = args.max_length
batch_size = args.batch_size
m = args.m
nb_epochs = args.nb_epochs
bert_init = args.bert_init
pretrained_bert_ckpt = args.pretrained_bert_ckpt
dataset = args.dataset
checkpoint_dir = args.checkpoint_dir
gcn_model = args.gcn_model
gcn_layers = args.gcn_layers
n_hidden = args.n_hidden
heads = args.heads
dropout = args.dropout
gcn_lr = args.gcn_lr
bert_lr = args.bert_lr
seed = args.seed
device_type = args.device
patience = args.patience
use_custom_test = args.use_custom_test
file_encoding = args.encoding
train_indices_path = args.train_indices
test_indices_path = args.test_indices
current_fold = args.current_fold
total_folds = args.total_folds

# Detect k-fold mode: both index files must be provided together
kfold_mode = (train_indices_path is not None) and (test_indices_path is not None)
if (train_indices_path is None) != (test_indices_path is None):
    raise ValueError('--train_indices and --test_indices must be provided together.')

# Set random seeds for reproducibility
import random
random.seed(seed)
np.random.seed(seed)
th.manual_seed(seed)
if th.cuda.is_available():
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
import dgl
dgl.seed(seed)

if checkpoint_dir is None:
    ckpt_dir = './checkpoint/{}_{}_{}'.format(bert_init, gcn_model, dataset)
else:
    ckpt_dir = checkpoint_dir
os.makedirs(ckpt_dir, exist_ok=True)
shutil.copy(os.path.basename(__file__), ckpt_dir)

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(message)s'))
sh.setLevel(logging.INFO)
fh = logging.FileHandler(filename=os.path.join(ckpt_dir, 'training.log'), mode='w')
fh.setFormatter(logging.Formatter('%(message)s'))
fh.setLevel(logging.INFO)
logger = logging.getLogger('training logger')
logger.addHandler(sh)
logger.addHandler(fh)
logger.setLevel(logging.INFO)

cpu = th.device('cpu')
gpu = th.device('cuda:0') if device_type == 'cuda' and th.cuda.is_available() else th.device('cpu')

# Override device if CUDA not available
if device_type == 'cuda' and not th.cuda.is_available():
    logger.warning('CUDA not available, using CPU instead')
    device_type = 'cpu'
    gpu = cpu

logger.info('arguments:')
logger.info(str(args))
logger.info('Random seed: {}'.format(seed))
logger.info('Device: {}'.format(device_type))
logger.info('File encoding: {}'.format(file_encoding))
logger.info('K-fold mode: {}'.format(kfold_mode))
if kfold_mode:
    logger.info(f'Current Fold: {current_fold}/{total_folds}')
logger.info('checkpoints will be saved in {}'.format(ckpt_dir))
# Model


# Data Preprocess
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset)
'''
adj: n*n sparse adjacency matrix
y_train, y_val, y_test: n*c matrices 
train_mask, val_mask, test_mask: n-d bool array
'''

# compute number of real train/val/test/word nodes and number of classes
nb_node = features.shape[0]
nb_train, nb_val, nb_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
nb_word = nb_node - nb_train - nb_val - nb_test
nb_class = y_train.shape[1]
# orig_nb_test reflects the corpus file layout (fixed across folds),
# NOT the fold's test size. Word nodes sit between train/val docs and
# test docs in the file. Do NOT replace this with nb_test after the
# kfold override or the word-node zero padding will be inserted at
# the wrong position.
orig_nb_test = int(nb_test)

# ── K-fold mask override ─────────────────────────────────────────────────────────
# Replace train/val/test masks with the fold indices supplied by the orchestrator.
# The graph (adj, features) is untouched — it was built from ALL documents.
#
# load_corpus() returns masks indexed over the full node space:
#   [0 .. nb_train-1]                         = train doc nodes
#   [nb_train .. nb_train+nb_val-1]            = val doc nodes
#   [nb_train+nb_val .. nb_train+nb_val+nb_word-1] = word nodes
#   [nb_train+nb_val+nb_word .. nb_node-1]     = test doc nodes
#
# When --train_indices / --test_indices are provided we rebuild those masks
# from scratch, using the raw document indices (positions in the original
# corpus file) supplied by run_kfold_twitter.py.

if kfold_mode:
    def _read_index_file(path):
        with open(path, 'r', encoding='utf-8') as fh:
            return [int(line.strip()) for line in fh if line.strip()]

    fold_train_idx = _read_index_file(train_indices_path)
    fold_test_idx  = _read_index_file(test_indices_path)

    # The fold splitter operates on the *labeled* document indices (positions
    # in the original corpus / .txt file).  load_corpus() maps those positions
    # into the shuffled node space via the .train.index file.
    # The graph was built from all documents, so the shuffled order in
    # data/<dataset>_shuffle.txt matches the original row order exactly
    # (build_graph.py writes train first, then test, but here we rebuilt the
    # full document pool without an explicit test split).  We therefore treat
    # the fold indices as direct node indices into the graph.

    labels = y_train + y_val + y_test   # full label matrix from load_corpus

    # 90/10 val split taken from the fold's training set
    from sklearn.model_selection import StratifiedShuffleSplit

    fold_train_labels = [labels[i].argmax() for i in fold_train_idx]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    tr_sub, val_sub = next(sss.split(fold_train_idx, fold_train_labels))
    real_train_idx = [fold_train_idx[i] for i in tr_sub]
    fold_val_idx   = [fold_train_idx[i] for i in val_sub]

    fold_train_size = len(fold_train_idx)
    fold_val_size   = len(fold_val_idx)
    fold_test_size  = len(fold_test_idx)

    n_nodes = nb_node
    train_mask = sample_mask(real_train_idx,  n_nodes)
    val_mask   = sample_mask(fold_val_idx,    n_nodes)
    test_mask  = sample_mask(fold_test_idx,   n_nodes)

    y_train = np.zeros(labels.shape)
    y_val   = np.zeros(labels.shape)
    y_test  = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val  [val_mask,   :] = labels[val_mask,   :]
    y_test [test_mask,  :] = labels[test_mask,  :]

    nb_train = train_mask.sum()
    nb_val   = val_mask.sum()
    nb_test  = test_mask.sum()

    logger.info('K-fold mask override applied:')
    logger.info('  Fold train (real) : {}'.format(len(real_train_idx)))
    logger.info('  Fold val          : {}'.format(fold_val_size))
    logger.info('  Fold test         : {}'.format(fold_test_size))

if gcn_model == 'gcn':
    model = BertGCN(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    n_hidden=n_hidden, dropout=dropout)
else:
    model = BertGAT(nb_class=nb_class, pretrained_model=bert_init, m=m, gcn_layers=gcn_layers,
                    heads=heads, n_hidden=n_hidden, dropout=dropout)


if pretrained_bert_ckpt is not None:
    ckpt = th.load(pretrained_bert_ckpt, map_location=gpu)
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])


# load documents and compute input encodings
corpse_file = './data/corpus/' + dataset +'_shuffle.txt'
try:
    # Try with specified encoding first
    with open(corpse_file, 'r', encoding=file_encoding) as f:
        text = f.read()
        text = text.replace('\\', '')
        text = text.split('\n')
except UnicodeDecodeError:
    # Fall back to different encodings if specified encoding fails
    logger.warning(f"Failed to read with {file_encoding} encoding. Trying fallback encodings...")
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    if file_encoding in encodings_to_try:
        encodings_to_try.remove(file_encoding)
    
    success = False
    for enc in encodings_to_try:
        try:
            with open(corpse_file, 'r', encoding=enc) as f:
                text = f.read()
                text = text.replace('\\', '')
                text = text.split('\n')
            logger.info(f"Successfully read file with {enc} encoding")
            success = True
            break
        except UnicodeDecodeError:
            continue
    
    if not success:
        # Last resort: use binary mode and ignore errors
        logger.warning("All encodings failed. Using binary mode with error ignoring...")
        with open(corpse_file, 'rb') as f:
            raw_data = f.read()
            text = raw_data.decode('utf-8', errors='ignore')
            text = text.replace('\\', '')
            text = text.split('\n')

def encode_input(text, tokenizer):
    input = tokenizer(text, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
#     print(input.keys())
    return input.input_ids, input.attention_mask


input_ids, attention_mask = encode_input(text, model.tokenizer)
input_ids = th.cat([input_ids[:-orig_nb_test], th.zeros((nb_word, max_length), dtype=th.long), input_ids[-orig_nb_test:]])
attention_mask = th.cat([attention_mask[:-orig_nb_test], th.zeros((nb_word, max_length), dtype=th.long), attention_mask[-orig_nb_test:]])

# transform one-hot label to class ID for pytorch computation
y = y_train + y_test + y_val
y_train = y_train.argmax(axis=1)
y = y.argmax(axis=1)

# document mask used for update feature
doc_mask  = train_mask + val_mask + test_mask

# build DGL Graph
adj_norm = normalize_adj(adj + sp.eye(adj.shape[0]))
g = dgl.from_scipy(adj_norm.astype('float32'), eweight_name='edge_weight')
g.ndata['input_ids'], g.ndata['attention_mask'] = input_ids, attention_mask
g.ndata['label'], g.ndata['train'], g.ndata['val'], g.ndata['test'] = \
    th.LongTensor(y), th.FloatTensor(train_mask), th.FloatTensor(val_mask), th.FloatTensor(test_mask)
g.ndata['label_train'] = th.LongTensor(y_train)
g.ndata['cls_feats'] = th.zeros((nb_node, model.feat_dim))

logger.info('graph information:')
logger.info(str(g))

# ── Index loaders ─────────────────────────────────────────────────────────────
# In k-fold mode the "train / val / test" node indices are no longer contiguous
# blocks, so we cannot use simple arange slices.  We derive them from the masks.

if kfold_mode:
    train_node_idx = th.where(th.BoolTensor(train_mask))[0]
    val_node_idx   = th.where(th.BoolTensor(val_mask))[0]
    test_node_idx  = th.where(th.BoolTensor(test_mask))[0]
    doc_node_idx   = th.where(th.BoolTensor(doc_mask))[0]

    train_idx = Data.TensorDataset(train_node_idx)
    val_idx   = Data.TensorDataset(val_node_idx)
    test_idx  = Data.TensorDataset(test_node_idx)
    doc_idx   = Data.TensorDataset(doc_node_idx)
else:
    train_idx = Data.TensorDataset(th.arange(0, nb_train, dtype=th.long))
    val_idx   = Data.TensorDataset(th.arange(nb_train, nb_train + nb_val, dtype=th.long))
    test_idx  = Data.TensorDataset(th.arange(nb_node - nb_test, nb_node, dtype=th.long))
    doc_idx   = Data.ConcatDataset([train_idx, val_idx, test_idx])

idx_loader_train = Data.DataLoader(train_idx, batch_size=batch_size, shuffle=True)
idx_loader_val = Data.DataLoader(val_idx, batch_size=batch_size)
idx_loader_test = Data.DataLoader(test_idx, batch_size=batch_size)
idx_loader = Data.DataLoader(doc_idx, batch_size=batch_size, shuffle=True)

# Load custom test set if specified (for semeval3a)
custom_test_loader = None
custom_test_data = None
if use_custom_test and dataset == 'semeval3a':
    logger.info("Loading custom test set for semeval3a...")
    import pandas as pd
    custom_test_df = pd.read_csv('data/semeval_2018_3a_custom_test.csv')
    custom_texts = custom_test_df['sentence'].tolist()
    custom_labels = custom_test_df['sentiment'].tolist()
    
    # Tokenize custom test texts
    custom_tokenizer_output = model.tokenizer(custom_texts, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
    custom_input_ids = custom_tokenizer_output['input_ids']
    custom_attention_mask = custom_tokenizer_output['attention_mask']
    custom_labels_tensor = th.LongTensor(custom_labels)
    
    custom_test_data = {
        'input_ids': custom_input_ids,
        'attention_mask': custom_attention_mask,
        'labels': custom_labels_tensor,
        'texts': custom_texts
    }
    
    # Create dataloader
    custom_test_dataset = Data.TensorDataset(th.arange(len(custom_texts), dtype=th.long))
    custom_test_loader = Data.DataLoader(custom_test_dataset, batch_size=batch_size)
    logger.info(f"Custom test set loaded: {len(custom_texts)} samples")

# Training
def update_feature():
    global model, g, doc_mask
    # no gradient needed, uses a large batchsize to speed up the process
    dataloader = Data.DataLoader(
        Data.TensorDataset(g.ndata['input_ids'][doc_mask], g.ndata['attention_mask'][doc_mask]),
        batch_size=1024
    )
    with th.no_grad():
        model = model.to(gpu)
        model.eval()
        cls_list = []
        for batch in dataloader:
            input_ids, attention_mask = [x.to(gpu) for x in batch]
            output = model.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            cls_list.append(output.cpu())
        cls_feat = th.cat(cls_list, axis=0)
    g = g.to(cpu)
    g.ndata['cls_feats'][doc_mask] = cls_feat
    return g


optimizer = th.optim.Adam([
        {'params': model.bert_model.parameters(), 'lr': bert_lr},
        {'params': model.classifier.parameters(), 'lr': bert_lr},
        {'params': model.gcn.parameters(), 'lr': gcn_lr},
    ], lr=1e-3
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma=0.1)


def train_step(engine, batch):
    global model, g, optimizer
    model.train()
    model = model.to(gpu)
    g = g.to(gpu)
    optimizer.zero_grad()
    (idx, ) = [x.to(gpu) for x in batch]
    optimizer.zero_grad()
    train_mask = g.ndata['train'][idx].type(th.BoolTensor)
    y_pred = model(g, idx)[train_mask]
    y_true = g.ndata['label_train'][idx][train_mask]
    loss = F.nll_loss(y_pred, y_true)
    loss.backward()
    optimizer.step()
    g.ndata['cls_feats'].detach_()
    train_loss = loss.item()
    with th.no_grad():
        if train_mask.sum() > 0:
            y_true = y_true.detach().cpu()
            y_pred = y_pred.argmax(axis=1).detach().cpu()
            train_acc = accuracy_score(y_true, y_pred)
        else:
            train_acc = 1
    return train_loss, train_acc


trainer = Engine(train_step)


@trainer.on(Events.EPOCH_COMPLETED)
def reset_graph(trainer):
    scheduler.step()
    update_feature()
    th.cuda.empty_cache()


def test_step(engine, batch):
    global model, g
    with th.no_grad():
        model.eval()
        model = model.to(gpu)
        g = g.to(gpu)
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        return y_pred, y_true


evaluator = Engine(test_step)
metrics={
    'acc': Accuracy(),
    'nll': Loss(th.nn.NLLLoss()),
    'f1_weighted': F1(average='weighted')
}
for n, f in metrics.items():
    f.attach(evaluator, n)


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(idx_loader_train)
    metrics = evaluator.state.metrics
    train_acc, train_nll, train_f1 = metrics["acc"], metrics["nll"], metrics["f1_weighted"]
    evaluator.run(idx_loader_val)
    metrics = evaluator.state.metrics
    val_acc, val_nll, val_f1 = metrics["acc"], metrics["nll"], metrics["f1_weighted"]
    evaluator.run(idx_loader_test)
    metrics = evaluator.state.metrics
    test_acc, test_nll, test_f1 = metrics["acc"], metrics["nll"], metrics["f1_weighted"]
    
    logger.info(
        "\rEpoch: {}  Train f1: {:.4f} loss: {:.4f}  Val f1: {:.4f} loss: {:.4f}  Test f1: {:.4f} loss: {:.4f}"
        .format(trainer.state.epoch, train_f1, train_nll, val_f1, val_nll, test_f1, test_nll)
    )
    
    # Early stopping logic
    if val_f1 > log_training_results.best_val_f1:
        logger.info("New checkpoint")
        th.save(
            {
                'bert_model': model.bert_model.state_dict(),
                'classifier': model.classifier.state_dict(),
                'gcn': model.gcn.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': trainer.state.epoch,
            },
            os.path.join(
                ckpt_dir, 'checkpoint.pth'
            )
        )
        log_training_results.best_val_f1 = val_f1
        log_training_results.patience_counter = 0
    else:
        log_training_results.patience_counter += 1
        logger.info(f"Patience: {log_training_results.patience_counter}/{patience}")
        
        if log_training_results.patience_counter >= patience:
            logger.info(f"Early stopping triggered after {patience} epochs without improvement")
            trainer.terminate()


log_training_results.best_val_acc = 0
log_training_results.patience_counter = 0
g = update_feature()

pbar = ProgressBar(persist=True, dynamic_ncols=True)
pbar.attach(trainer, 
             output_transform=lambda x: {
                 'loss': x[0], 
                 'acc': x[1],
                 'fold': f'{current_fold}/{total_folds}' if kfold_mode else ''
             })

trainer.run(idx_loader, max_epochs=nb_epochs)

# Final test evaluation with detailed metrics
logger.info("\n" + "="*80)
logger.info("FINAL TEST EVALUATION")
logger.info("="*80)

# Load best model
checkpoint_path = os.path.join(ckpt_dir, 'checkpoint.pth')
if os.path.exists(checkpoint_path):
    checkpoint = th.load(checkpoint_path)
    model.bert_model.load_state_dict(checkpoint['bert_model'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    model.gcn.load_state_dict(checkpoint['gcn'])
    logger.info("Loaded best checkpoint from epoch {}".format(checkpoint['epoch']))
else:
    logger.info("No checkpoint found, using final model")

# Evaluate on test set
model.eval()
model = model.to(gpu)
g = g.to(gpu)

all_preds = []
all_labels = []

with th.no_grad():
    for batch in idx_loader_test:
        (idx, ) = [x.to(gpu) for x in batch]
        y_pred = model(g, idx)
        y_true = g.ndata['label'][idx]
        
        all_preds.append(y_pred.argmax(axis=1).cpu().numpy())
        all_labels.append(y_true.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Calculate metrics
test_accuracy = accuracy_score(all_labels, all_preds)
test_f1_macro = f1_score(all_labels, all_preds, average='macro')
test_f1_per_class = f1_score(all_labels, all_preds, average=None)

# Determine class names
if dataset in ['isarcasm']:
    class_names = ['not_sarcastic', 'sarcastic']
elif dataset in ['semeval3a']:
    class_names = ['not_ironic', 'ironic']
elif dataset in ['twitter']:
    class_names = ['positive', 'negative', 'neutral']
else:
    class_names = [str(i) for i in range(nb_class)]

logger.info("\nTest Set Results:")
logger.info("  Accuracy: {:.4f}".format(test_accuracy))
logger.info("  F1 Macro: {:.4f}".format(test_f1_macro))

if dataset in ['isarcasm']:
    logger.info("  F1 Not Sarcastic: {:.4f}".format(test_f1_per_class[0]))
    logger.info("  F1 Sarcastic: {:.4f}".format(test_f1_per_class[1]))
elif dataset in ['semeval3a']:
    logger.info("  F1 Not Ironic: {:.4f}".format(test_f1_per_class[0]))
    logger.info("  F1 Ironic: {:.4f}".format(test_f1_per_class[1]))
elif dataset in ['twitter']:
    logger.info("  F1 Positive: {:.4f}".format(test_f1_per_class[0]))
    logger.info("  F1 Negative: {:.4f}".format(test_f1_per_class[1]))
    logger.info("  F1 Neutral: {:.4f}".format(test_f1_per_class[2]))
else:
    for i, f1 in enumerate(test_f1_per_class):
        logger.info("  F1 Class {}: {:.4f}".format(i, f1))

logger.info("\nDetailed Classification Report:")
logger.info("\n" + classification_report(all_labels, all_preds,
                                         target_names=class_names,
                                         digits=4))
logger.info("="*80)

# ── NEW: Save confusion matrix ────────────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

# Pretty-print to log
logger.info("\nConfusion Matrix (rows=true, cols=predicted):")
header = "              " + "  ".join(f"{n:>14}" for n in class_names)
logger.info(header)
for i, row in enumerate(cm):
    row_str = f"{class_names[i]:>14}  " + "  ".join(f"{v:>14}" for v in row)
    logger.info(row_str)
logger.info("")

# Save as JSON (machine-readable, easy to load later for aggregation)
cm_path = os.path.join(ckpt_dir, 'confusion_matrix.json')
with open(cm_path, 'w', encoding='utf-8') as f:
    json.dump({
        'confusion_matrix': cm.tolist(),
        'class_names': class_names,
        'dataset': dataset,
        'seed': seed,
        'fold': current_fold if kfold_mode else None,
        'accuracy': float(test_accuracy),
        'f1_macro': float(test_f1_macro),
    }, f, indent=2)
logger.info("Confusion matrix saved to: {}".format(cm_path))
# ── END confusion matrix ──────────────────────────────────────────────────────

# Evaluate on custom test set if available
if custom_test_loader is not None and custom_test_data is not None:
    logger.info("\n" + "="*80)
    logger.info("CUSTOM TEST SET EVALUATION (semeval3a)")
    logger.info("="*80)
    
    model.eval()
    model = model.to(gpu)
    
    custom_preds = []
    custom_labels = custom_test_data['labels'].numpy()
    
    with th.no_grad():
        # Process in batches
        for i in range(0, len(custom_test_data['input_ids']), batch_size):
            batch_input_ids = custom_test_data['input_ids'][i:i+batch_size].to(gpu)
            batch_attention_mask = custom_test_data['attention_mask'][i:i+batch_size].to(gpu)
            
            # Get BERT features
            bert_output = model.bert_model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)[0][:, 0]
            
            # Get predictions (only using BERT classifier, no graph for custom data)
            logits = model.classifier(bert_output)
            preds = logits.argmax(axis=1).cpu().numpy()
            custom_preds.extend(preds)
    
    custom_preds = np.array(custom_preds)
    
    # Calculate metrics
    custom_accuracy = accuracy_score(custom_labels, custom_preds)
    custom_f1_macro = f1_score(custom_labels, custom_preds, average='macro')
    custom_f1_per_class = f1_score(custom_labels, custom_preds, average=None)
    
    logger.info("\nCustom Test Set Results:")
    logger.info("  Accuracy: {:.4f}".format(custom_accuracy))
    logger.info("  F1 Macro: {:.4f}".format(custom_f1_macro))
    
    if dataset == 'semeval3a':
        class_names = ['not_ironic', 'ironic']
        logger.info("  F1 Not Ironic: {:.4f}".format(custom_f1_per_class[0]))
        logger.info("  F1 Ironic: {:.4f}".format(custom_f1_per_class[1]))
    
    logger.info("\nDetailed Classification Report (Custom Test):")
    logger.info("\n" + classification_report(custom_labels, custom_preds, 
                                            target_names=class_names, 
                                            digits=4))
    logger.info("="*80)

# Save final results to file
results_file = os.path.join(ckpt_dir, 'final_results.txt')
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("FINAL TEST RESULTS\n")
    f.write("="*80 + "\n")
    f.write("Dataset: {}\n".format(dataset))
    f.write("Seed: {}\n".format(seed))
    f.write("Device: {}\n".format(device_type))
    f.write("File encoding: {}\n".format(file_encoding))
    f.write("K-fold mode: {}\n".format(kfold_mode))
    if kfold_mode:
        f.write("Train indices: {}\n".format(train_indices_path))
        f.write("Test indices:  {}\n".format(test_indices_path))
    f.write("\n")
    f.write("Test Accuracy: {:.4f}\n".format(test_accuracy))
    f.write("Test F1 Macro: {:.4f}\n".format(test_f1_macro))
    f.write("\n")
    if dataset in ['isarcasm']:
        f.write("F1 Not Sarcastic: {:.4f}\n".format(test_f1_per_class[0]))
        f.write("F1 Sarcastic: {:.4f}\n".format(test_f1_per_class[1]))
    elif dataset in ['semeval3a']:
        f.write("F1 Not Ironic: {:.4f}\n".format(test_f1_per_class[0]))
        f.write("F1 Ironic: {:.4f}\n".format(test_f1_per_class[1]))
    elif dataset in ['twitter']:
        f.write("F1 Positive: {:.4f}\n".format(test_f1_per_class[0]))
        f.write("F1 Negative: {:.4f}\n".format(test_f1_per_class[1]))
        f.write("F1 Neutral: {:.4f}\n".format(test_f1_per_class[2]))
    else:
        for i, f1 in enumerate(test_f1_per_class):
            f.write("F1 Class {}: {:.4f}\n".format(i, f1))
    f.write("\n")
    f.write(classification_report(all_labels, all_preds,
                                  target_names=class_names,
                                  digits=4))
    f.write("\n")
    # Also write confusion matrix to final_results.txt for human readability
    f.write("Confusion Matrix (rows=true, cols=predicted):\n")
    f.write("              " + "  ".join(f"{n:>14}" for n in class_names) + "\n")
    for i, row in enumerate(cm):
        f.write(f"{class_names[i]:>14}  " + "  ".join(f"{v:>14}" for v in row) + "\n")

logger.info("Results saved to: {}".format(results_file))