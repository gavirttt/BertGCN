"""
run_kfold_twitter.py  (Transductive k-fold)

The graph is built ONCE from all documents before any CV loop starts.
Per-fold, only the train/test masks change — the graph structure is unchanged.

Pipeline:
  1. prepare_twt_dataset.py  — run once manually (not called here)
  2. build_graph.py twitter  — run once manually (not called here)
  3. This script:
       a. Loads labeled doc indices from data/twitter.txt
       b. Splits them into k folds (StratifiedGroupKFold by conversation ID,
          or plain StratifiedKFold with --no_keep_conversations)
       c. For each fold, checks if a persistent fold-level BERT checkpoint
          already exists before finetuning. If it does, reuses it; otherwise
          finetunes BERT on that fold's train set and saves to a stable
          deterministic directory so subsequent runs can skip this step.
       d. Writes two temp index files and calls train_bert_gcn.py
          with --train_indices / --test_indices so only the masks change
       e. Aggregates metrics across folds and seeds

Fold BERT checkpoint directory layout:
  ./checkpoint/bert_fold{fold_id}_seed{seed}_k{k}_{bert_init_slug}_{dataset}/
    checkpoint.pth   — best BERT weights for this fold (persists across runs)
    training.log

Usage:
    # Single seed, 5-fold, conversation-aware (default)
    python run_kfold_twitter.py --k 5 --seed 42 --nb_epochs 50

    # With pretrained BERT checkpoint
    python run_kfold_twitter.py --k 5 --seed 42 --nb_epochs 50 \
        --pretrained_bert_ckpt ./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth

    # Multiple seeds, GPU
    python run_kfold_twitter.py --k 5 --seeds 42 43 44 --nb_epochs 50 --device cuda

    # Ignore conversation structure
    python run_kfold_twitter.py --k 5 --seed 42 --no_keep_conversations

Prerequisites (run once before this script):
    python prepare_twt_dataset.py \
        --csv data/tweets_labeled_set.csv \
        --unlabeled_csv data/tweets_unlabeled_set.csv
    python build_graph.py twitter --seed 42
"""

import argparse
import os
import pickle
import re
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET = 'twitter'


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_labeled_df(data_dir: str = 'data') -> tuple:
    """
    Returns (df, conversation_map) where df has columns:
        doc_index  int  — row number in data/twitter.txt
        label      str  — sentiment label
        conv_id    str  — conversation group key
    Only labeled (non-unlabeled) rows are included.
    """
    rows = []
    with open(f'{data_dir}/{DATASET}.txt', 'r', encoding='utf-8') as fh:
        for i, line in enumerate(fh):
            parts = line.strip().split('\t')
            if len(parts) == 3:
                rows.append({'doc_index': i, 'split': parts[1], 'label': parts[2]})

    df = pd.DataFrame(rows)
    df = df[df['label'] != 'unlabeled'].reset_index(drop=True)

    print(f'Labeled documents for CV : {len(df)}')
    print(f'Label distribution:\n{df["label"].value_counts().to_string()}')

    conv_path = f'{data_dir}/{DATASET}_conversations.pkl'
    conversation_map = {}
    if os.path.exists(conv_path):
        with open(conv_path, 'rb') as fh:
            conversation_map = pickle.load(fh)

    doc_to_conv = {}
    for conv_id, doc_ids in conversation_map.items():
        for did in doc_ids:
            doc_to_conv[did] = str(conv_id)

    df['conv_id'] = df['doc_index'].map(doc_to_conv).fillna(
        'solo_' + df['doc_index'].astype(str))

    return df, conversation_map


def _group_kfold(df: pd.DataFrame, k: int, seed: int):
    """StratifiedGroupKFold — keeps every conversation inside one fold."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    X, y, g = df['doc_index'].values, df['label'].values, df['conv_id'].values
    for fold_id, (tr, te) in enumerate(sgkf.split(X, y, g)):
        yield fold_id, X[tr].tolist(), X[te].tolist()


def _stratified_kfold(df: pd.DataFrame, k: int, seed: int):
    """Plain StratifiedKFold — ignores conversation boundaries."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    X, y = df['doc_index'].values, df['label'].values
    for fold_id, (tr, te) in enumerate(skf.split(X, y)):
        yield fold_id, X[tr].tolist(), X[te].tolist()


def _write_index_file(indices: list, path: str):
    with open(path, 'w') as fh:
        fh.write('\n'.join(str(i) for i in indices))


def _run(cmd: list, desc: str) -> bool:
    print(f'\n{"─"*60}')
    print(f'  {desc}')
    print(f'  $ {" ".join(cmd)}')
    print(f'{"─"*60}')
    ok = subprocess.run(cmd).returncode == 0
    print(f'  {"✓" if ok else "✗"}  {desc}')
    return ok


def _parse_results(path: str) -> dict:
    metrics = {}
    if not os.path.exists(path):
        return metrics
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            for pattern, key in [
                (r'Test Accuracy:\s+([\d.]+)', 'accuracy'),
                (r'Test F1 Macro:\s+([\d.]+)',  'f1_macro'),
            ]:
                m = re.match(pattern, line)
                if m:
                    metrics[key] = float(m.group(1))
            m = re.match(r'F1 (.+?):\s+([\d.]+)', line)
            if m:
                metrics['f1_' + m.group(1).lower().replace(' ', '_')] = float(m.group(2))
    return metrics

def _build_corpus_to_node_map(data_dir: str, dataset: str) -> dict:
    """
    Reads data/<dataset>_shuffle.txt to map original corpus row index
    → shuffled graph node position. build_graph.py writes this file
    with doc names like 'doc_42' whose number is the original row index.
    """
    
    with open(f'{data_dir}/{dataset}.train.index', 'r', encoding='utf-8') as fh:
        train_size = len([l for l in fh if l.strip()])

    with open(f'{data_dir}/corpus/{dataset}_vocab.txt', 'r', encoding='utf-8') as fh:
        vocab_size = len([l for l in fh if l.strip()])

    corpus_to_node = {}
    with open(f'{data_dir}/{dataset}_shuffle.txt', 'r', encoding='utf-8') as fh:
        for node_idx, line in enumerate(fh):
            parts = line.strip().split('\t')
            if parts and parts[0]:
                orig_idx = int(parts[0].split('_')[1])
                if node_idx < train_size:
                    corpus_to_node[orig_idx] = node_idx
                else:
                    corpus_to_node[orig_idx] = node_idx + vocab_size
    return corpus_to_node

def _log_class_distribution(indices, label_df, split_name, logger=None):
    """
    Log class distribution for a set of indices.
    
    Args:
        indices: List of document indices (original corpus indices, not graph node indices)
        label_df: DataFrame with 'label' column and index access via .loc
        split_name: String name for this split (e.g., 'Train', 'Val', 'Test')
        logger: Optional logger function (defaults to print)
    """
    labels = label_df.loc[indices, 'label'] if isinstance(indices, list) else label_df.iloc[indices]['label']
    label_counts = labels.value_counts()
    total = len(indices)
    
    output = [f"\n{split_name} Set Class Distribution (n={total}):"]
    for label, count in label_counts.items():
        pct = (count / total) * 100
        output.append(f"  {label:12s}: {count:6d} ({pct:5.2f}%)")
    
    output_str = '\n'.join(output)
    if logger:
        logger(output_str)
    else:
        print(output_str)
    
    return label_counts


def _fold_bert_ckpt_dir(fold_id: int, seed: int, k: int, bert_init: str) -> str:
    """
    Returns the deterministic, persistent directory for a fold's finetuned
    BERT checkpoint.  The path is stable across runs so subsequent calls
    can detect and reuse existing checkpoints.

    Layout:
        ./checkpoint/finetuned/bert_fold{fold_id}_seed{seed}_k{k}_{bert_slug}_{dataset}/
    """
    bert_slug = bert_init.replace('/', '_').replace('\\', '_')
    return os.path.join(
        './checkpoint/finetuned',
        f'bert_fold{fold_id}_seed{seed}_k{k}_{bert_slug}_{DATASET}'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed CV run
# ─────────────────────────────────────────────────────────────────────────────

def run_kfold(
    k: int,
    seed: int,
    m: float,
    nb_epochs: int,
    device: str,
    gcn_model: str,
    batch_size: int,
    bert_init: str,
    keep_conversations: bool,
    data_dir: str,
    pretrained_bert_ckpt: str = None,
) -> list:

    print(f'\n{"="*70}')
    print(f'  TWITTER K-FOLD |  k={k}  seed={seed}  '
          f'conv_aware={keep_conversations}')
    print(f'  Graph is NOT rebuilt per fold — only masks change.')
    if pretrained_bert_ckpt:
        print(f'  Pretrained BERT: {pretrained_bert_ckpt}')
    else:
        print(f'  Pretrained BERT: None (training from scratch)')
    print(f'{"="*70}')

    df, conversation_map = _load_labeled_df(data_dir)

    splitter = (
        _group_kfold(df, k, seed)
        if keep_conversations
        else _stratified_kfold(df, k, seed)
    )

    fold_metrics = []
    corpus_to_node = _build_corpus_to_node_map(data_dir, DATASET)
    # ---------------------Sanity check-------------------------
    with open(f'{data_dir}/{DATASET}.train.index', 'r', encoding='utf-8') as fh:
        train_size = len([l for l in fh if l.strip()])
    with open(f'{data_dir}/corpus/{DATASET}_vocab.txt', 'r', encoding='utf-8') as fh:
        vocab_size = len([l for l in fh if l.strip()])
    train_nodes = [v for v in corpus_to_node.values() if v < train_size]
    test_nodes = [v for v in corpus_to_node.values() if v >= train_size + vocab_size]
    print(f"Train-position docs: {len(train_nodes)}, Test-position docs: {len(test_nodes)}")
    # ----------------------------------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        for fold_id, train_doc_idx, test_doc_idx in tqdm(splitter, desc=f'Seed {seed} K-Fold CV', unit='fold', leave=True):

            print(f'\n{"─"*60}')
            print(f'  FOLD {fold_id + 1}/{k}  '
                  f'train={len(train_doc_idx)}  test={len(test_doc_idx)}')

            idx_set = df.set_index('doc_index')

            # === CLASS DISTRIBUTION LOGGING ===
            print(f'\n{"="*60}')
            print(f'  CLASS DISTRIBUTION FOR FOLD {fold_id + 1}/{k}')
            print(f'{"="*60}')
            
            # Log full fold train set (pre-validation split)
            _log_class_distribution(train_doc_idx, idx_set, "Full Train (pre-val)")
            print()
            
            # Create val split from train_doc_idx
            fold_train_labels = [idx_set.loc[i, 'label'] for i in train_doc_idx]
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
            tr_sub, val_sub = next(sss.split(train_doc_idx, fold_train_labels))
            real_train_idx = [train_doc_idx[i] for i in tr_sub]
            fold_val_idx   = [train_doc_idx[i] for i in val_sub]
            
            # Log actual train (post-val) and validation sets
            _log_class_distribution(real_train_idx, idx_set, "Real Train (post-val)")
            _log_class_distribution(fold_val_idx, idx_set, "Validation")
            _log_class_distribution(test_doc_idx, idx_set, "Test")
            
            # Log class imbalance ratios
            train_counts = idx_set.loc[real_train_idx, 'label'].value_counts()
            val_counts = idx_set.loc[fold_val_idx, 'label'].value_counts()
            test_counts = idx_set.loc[test_doc_idx, 'label'].value_counts()
            
            print(f"\n  Imbalance Ratios (max/min):")
            print(f"    Train: {train_counts.max() / train_counts.min():.2f}")
            print(f"    Val:   {val_counts.max() / val_counts.min():.2f}")
            print(f"    Test:  {test_counts.max() / test_counts.min():.2f}")
            print(f'{"="*60}\n')
            # === END CLASS DISTRIBUTION LOGGING ===

            print('  Train labels:', dict(idx_set.loc[train_doc_idx, 'label'].value_counts()))
            print('  Test  labels:', dict(idx_set.loc[test_doc_idx,  'label'].value_counts()))

            test_label_counts = idx_set.loc[test_doc_idx, 'label'].value_counts()
            if test_label_counts.min() < 5:
                print(f'  ⚠ Warning: fold {fold_id+1} has only '
                    f'{test_label_counts.min()} minority class test examples — '
                    f'F1 for that class will be unreliable')

            if keep_conversations and conversation_map:
                fold_set   = set(train_doc_idx) | set(test_doc_idx)
                split_conv = sum(
                    1 for docs in conversation_map.values()
                    if len(docs) > 1
                    and any(d in fold_set for d in docs)
                    and not all(d in fold_set for d in docs)
                )
                print(f'  Conversations cut at fold boundary: {split_conv}')

            train_file = os.path.join(tmpdir, f'train_fold{fold_id}.txt')
            test_file  = os.path.join(tmpdir, f'test_fold{fold_id}.txt')
            _write_index_file([corpus_to_node[i] for i in train_doc_idx], train_file)
            _write_index_file([corpus_to_node[i] for i in test_doc_idx],  test_file)

            # ── Persistent fold BERT checkpoint ───────────────────────────
            # Use a stable, deterministic directory so the checkpoint survives
            # across runs. Subsequent runs skip finetuning if it already exists.
            bert_ckpt_dir = _fold_bert_ckpt_dir(fold_id, seed, k, bert_init)
            fold_bert_ckpt = os.path.join(bert_ckpt_dir, 'checkpoint.pth')

            if os.path.exists(fold_bert_ckpt):
                print(f'\n  ✓ Reusing existing fold BERT checkpoint:')
                print(f'    {fold_bert_ckpt}')
            else:
                print(f'\n  No fold BERT checkpoint found at:')
                print(f'    {fold_bert_ckpt}')
                print(f'  Finetuning BERT for fold {fold_id + 1}/{k} ...')

                fold_bert_train_file = os.path.join(tmpdir, f'bert_train_fold{fold_id}.txt')
                _write_index_file(train_doc_idx, fold_bert_train_file)

                os.makedirs(bert_ckpt_dir, exist_ok=True)
                cmd_finetune = [
                    sys.executable, 'finetune_bert.py',
                    '--dataset', DATASET,
                    '--bert_init', bert_init,
                    '--train_indices', fold_bert_train_file,
                    '--checkpoint_dir', bert_ckpt_dir,
                    '--nb_epochs', str(10),
                    '--bert_lr', '2e-5'
                ]
                _run(cmd_finetune, f'Finetune BERT — fold {fold_id + 1}/{k}')

                if not os.path.exists(fold_bert_ckpt):
                    print(f'  ✗ WARNING: finetune_bert.py did not produce a checkpoint at {fold_bert_ckpt}')
                    print(f'    Fold {fold_id + 1} will proceed without a pretrained BERT checkpoint.')

            ckpt_dir = (
                f'./checkpoint/finetuned/{DATASET}_fold{fold_id}_seed{seed}_{gcn_model}_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )

            # Pass --dataset twitter (full graph) plus fold index files.
            # train_bert_gcn.py loads ind.twitter.* unchanged and only
            # replaces the train/val/test masks with these indices.
            # Build the command, conditionally adding --pretrained_bert_ckpt
            cmd = [
                sys.executable, 'train_bert_gcn.py',
                '--dataset',        DATASET,
                '--train_indices',  train_file,
                '--test_indices',   test_file,
                '--seed',           str(seed),
                '--m',              str(m),
                '--nb_epochs',      str(nb_epochs),
                '--gcn_model',      gcn_model,
                '--device',         device,
                '--batch_size',     str(batch_size),
                '--bert_init',      bert_init,
                '--checkpoint_dir', ckpt_dir,
                '--current_fold',   str(fold_id + 1),
                '--total_folds',    str(k),
                '--pretrained_bert_ckpt', fold_bert_ckpt
            ]

            ok = _run(cmd, f'Train — fold {fold_id + 1}/{k}')

            if not ok:
                fold_metrics.append({})
                continue

            metrics = _parse_results(os.path.join(ckpt_dir, 'final_results.txt'))
            metrics['fold']     = fold_id + 1
            metrics['ckpt_dir'] = ckpt_dir
            fold_metrics.append(metrics)
            print(f'  Fold {fold_id + 1} → {metrics}')

    return fold_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation + summary
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate(all_metrics: list, label: str = ''):
    valid = [m for m in all_metrics if 'accuracy' in m]
    if not valid:
        print('  No valid fold results to aggregate.')
        return
    print(f'\n{"="*70}')
    print(f'  AGGREGATED RESULTS  {label}  '
          f'({len(valid)}/{len(all_metrics)} folds)')
    print(f'{"="*70}')
    for key in ['accuracy', 'f1_macro']:
        vals = [m[key] for m in valid if key in m]
        if vals:
            print(f'  {key:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}'
                  f'  {[round(v, 4) for v in vals]}')
    for key in [k for k in valid[0] if k.startswith('f1_') and k != 'f1_macro']:
        vals = [m[key] for m in valid if key in m]
        if vals:
            print(f'  {key:25s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}')
    print(f'{"="*70}')


def _write_summary(all_results: dict, path: str):
    lines = ['seed,fold,accuracy,f1_macro,checkpoint']
    for seed, folds in all_results.items():
        for m in folds:
            if not m:
                continue
            lines.append(','.join([
                str(seed),
                str(m.get('fold', '')),
                f'{m["accuracy"]:.4f}' if 'accuracy' in m else '',
                f'{m["f1_macro"]:.4f}' if 'f1_macro'  in m else '',
                m.get('ckpt_dir', ''),
            ]))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    print(f'\n✓ Summary written to: {path}')


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='K-fold CV for Twitter/BertGCN — (transductive)')

    parser.add_argument('--k',          type=int, default=5)
    parser.add_argument('--seed',       type=int, help='Single seed shorthand')
    parser.add_argument('--seeds',      type=int, nargs='+', default=[42])
    parser.add_argument('--m',          type=float, default=0.7, help='the factor balancing BERT and GCN prediction')
    parser.add_argument('--nb_epochs',  type=int, default=50)
    parser.add_argument('--device',     type=str, default='cpu',
                        choices=['cpu', 'cuda'])
    parser.add_argument('--gcn_model',  type=str, default='gcn',
                        choices=['gcn', 'gat'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bert_init',  type=str,
                        default='jcblaise/roberta-tagalog-base')
    parser.add_argument('--pretrained_bert_ckpt', type=str, default=None,
                        help='Path to pretrained BERT checkpoint from finetune_bert.py. '
                             'If provided and the file exists, BERT weights are '
                             'initialised from this checkpoint before each fold.')
    parser.add_argument('--keep_conversations', dest='keep_conversations',
                        action='store_true', default=True,
                        help='Use StratifiedGroupKFold (default)')
    parser.add_argument('--no_keep_conversations',
                        dest='keep_conversations', action='store_false',
                        help='Use plain StratifiedKFold')
    parser.add_argument('--data_dir',     type=str, default='data')
    parser.add_argument('--summary_file', type=str,
                        default='kfold_twitter_summary.csv')

    args = parser.parse_args()

    seeds = list(args.seeds)
    if args.seed is not None and args.seed not in seeds:
        seeds = [args.seed] + seeds

    # Resolve and validate the pretrained BERT checkpoint path once up front
    pretrained_bert_ckpt = args.pretrained_bert_ckpt
    if pretrained_bert_ckpt is not None:
        pretrained_bert_ckpt = os.path.abspath(pretrained_bert_ckpt)
        if not os.path.exists(pretrained_bert_ckpt):
            print(f'⚠ Warning: --pretrained_bert_ckpt not found at {pretrained_bert_ckpt}')
            print(f'  All folds will train BERT from scratch.')

    # Print the expected fold BERT checkpoint paths so the user can see
    # which ones will be reused vs finetuned on this run.
    print('╔═════════════════════╗')
    print('║  Twitter K-Fold CV  ║')
    print('╚═════════════════════╝')
    print(f'  k                  : {args.k}')
    print(f'  seeds              : {seeds}')
    print(f'  epochs             : {args.nb_epochs}')
    print(f'  device             : {args.device}')
    print(f'  gcn model          : {args.gcn_model}')
    print(f'  bert init          : {args.bert_init}')
    print(f'  pretrained ckpt    : {pretrained_bert_ckpt or "None (train from scratch)"}')
    print(f'  keep conversations : {args.keep_conversations}')
    print()

    # Preview fold BERT checkpoint status for all seeds/folds
    print('  Fold BERT checkpoint status:')
    for seed in seeds:
        for fold_id in range(args.k):
            ckpt_dir = _fold_bert_ckpt_dir(fold_id, seed, args.k, args.bert_init)
            ckpt_path = os.path.join(ckpt_dir, 'checkpoint.pth')
            status = '✓ exists (will reuse)' if os.path.exists(ckpt_path) else '✗ missing (will finetune)'
            print(f'    seed={seed} fold={fold_id}: {status}')
            print(f'      {ckpt_path}')
    print()

    print('  NOTE: The graph must already exist on disk.')
    print(f'        Expected: data/ind.{DATASET}.adj')

    adj_path = f'{args.data_dir}/ind.{DATASET}.adj'
    if not os.path.exists(adj_path):
        print(f'\n✗ Graph not found at {adj_path}')
        print(f'  Run first:  python build_graph.py {DATASET} --seed <seed>')
        sys.exit(1)
    print(f'  Graph found ✓\n')

    all_results = {}
    for seed in seeds:
        fold_metrics = run_kfold(
            k                  = args.k,
            seed               = seed,
            m                  = args.m,
            nb_epochs          = args.nb_epochs,
            device             = args.device,
            gcn_model          = args.gcn_model,
            batch_size         = args.batch_size,
            bert_init          = args.bert_init,
            keep_conversations = args.keep_conversations,
            data_dir           = args.data_dir,
            pretrained_bert_ckpt = pretrained_bert_ckpt,
        )
        all_results[seed] = fold_metrics
        _aggregate(fold_metrics, label=f'seed={seed}')

    if len(seeds) > 1:
        all_folds = [m for folds in all_results.values() for m in folds]
        _aggregate(all_folds, label=f'all seeds {seeds}')

    _write_summary(all_results, args.summary_file)
    print('\n✓ Done!')
    print(f'  Results : {args.summary_file}')
    print(f'  Models  : ./checkpoint/twitter_fold*_seed*/')


if __name__ == '__main__':
    main()