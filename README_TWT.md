# Twitter BertGCN Ablation Pipeline

## Overview

This pipeline runs an ablation study on a Twitter sentiment dataset using BertGCN. It investigates whether adding conversation-based graph edges improves performance with a fixed balance factor between BERT and GCN predictions.

Following Lin et al. (2021), who report that BertGCN achieves optimal performance at `m = 0.7`, we use this value as our pre-specified hyperparameter for all primary comparisons. A pretrained BERT checkpoint is shared across all conditions.

### What `m` controls

The final prediction is a weighted combination of BERT and GCN outputs:

```
pred = gcn_pred * m + cls_pred * (1 - m)
```

### Sentiment label mapping

| Integer | Label |
|---------|-------|
| `0` | Positive |
| `1` | Negative |
| `2` | Neutral |

---

## Prerequisites

### Environment setup

**1. Create and activate the conda environment:**

```bash
conda create -n BertGCN python=3.8 -y
conda activate BertGCN
```

**2. Install PyTorch and related packages (CUDA 11.8):**

```bash
pip install torch==2.2.1+cu118 torchaudio==2.2.1+cu118 torchvision==0.17.1+cu118 torchdata==0.7.1 \
    --index-url https://download.pytorch.org/whl/cu118
```

**3. Install remaining dependencies:**

```bash
pip install transformers datasets nltk scipy pytorch-ignite scikit-learn pydantic tqdm
pip install emoji>=2.8.0 html5lib>=1.1 wordcloud umap-learn==0.5.4 matplotlib
```

**4. Install DGL (CUDA 11.8):**

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
```

**5. Verify the installation:**

```bash
python -c "
import torch, torchdata, dgl
print(f'Torch version:        {torch.__version__}')
print(f'Torch CUDA available: {torch.cuda.is_available()}')
print(f'Torch CUDA version:   {torch.version.cuda}')
print(f'TorchData version:    {torchdata.__version__}')
print(f'DGL version:          {dgl.__version__}')
"
```

> **Note:** These instructions target CUDA 11.8. If your driver requires a different CUDA version, replace `cu118` with the appropriate tag (e.g. `cu121`) and find the matching DGL wheel at https://www.dgl.ai/pages/start.html.

### Required input files

Verify that the following are in the `data/` directory before running:

| File | Description |
|------|-------------|
| `data/tweets_labeled_set.csv` | Labeled tweets with columns: `pseudo_id`, `text`, `sentiment`, `pseudo_conversationId` |

---

## Pipeline Steps

The full pipeline has 6 steps. Each step and its purpose is described below.

### Step 1 — Prepare the dataset

```bash
python prepare_twt_dataset.py \
    --csv data/tweets_labeled_set.csv \
    --test_split_ratio 0.1
```

Reads the labeled CSV, cleans tweet text, and writes the following files to `data/`:

- `twitter.txt` — document index file (doc_id, split, label)
- `corpus/twitter.txt` — raw tweet text
- `corpus/twitter.clean.txt` — cleaned tweet text
- `twitter_conversations.pkl` — conversation grouping map

The labeled tweets are split into training (90%) and test (10%) sets. The graph builder will internally create a 90/10 validation split from the training data (i.e., 81% train, 9% validation, 10% test overall).

---

### Step 2 — Build graph (no conversation edges)

```bash
python build_graph.py twitter --seed 42 --no_conversation_edges
```

Builds the heterogeneous word-document graph using TF-IDF and PMI edge weights. With `--no_conversation_edges`, only word-document co-occurrence edges are included — no edges are added between tweets that belong to the same conversation thread.

Outputs to `data/`:
- `ind.twitter.adj` — adjacency matrix
- `ind.twitter.x`, `ind.twitter.y`, etc. — feature and label matrices
- `twitter_shuffle.txt`, `twitter.train.index` — shuffled document ordering

> **Note:** This step must come before `finetune_bert.py` because finetuning requires `corpus/twitter_shuffle.txt`, which is generated here.

---

### Step 3 — Finetune BERT

```bash
python finetune_bert.py \
    --dataset twitter \
    --bert_init jcblaise/roberta-tagalog-base \
    --bert_lr 2e-5 \
    --nb_epochs 10
```

Finetunes the BERT model on the Twitter training documents without any graph component. The checkpoint is saved to:

```
./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth
```

This checkpoint is **reused for all subsequent k-fold runs** across both graph conditions. You do not need to re-run this step when switching between the no-conversation-edges and with-conversation-edges conditions, since BERT finetuning does not use the graph structure.

> **Warning:** Re-running this step with different hyperparameters will overwrite the existing checkpoint since the directory is not timestamped.

---

### Step 4 — K-fold CV: no conversation edges

Run with fixed `m = 0.7`:

```bash
python run_kfold_twitter.py \
    --k 5 \
    --seed 42 \
    --m 0.7 \
    --nb_epochs 5 \
    --device cuda \
    --bert_init jcblaise/roberta-tagalog-base \
    --pretrained_bert_ckpt ./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth \
    --summary_file kfold_no_conv_edges_m0.7_summary.csv
```

---

### Step 5 — Build graph (with conversation edges)

```bash
python build_graph.py twitter --seed 42
```

Rebuilds the graph with conversation-based edges added on top of the word-document edges. Tweets belonging to the same `pseudo_conversationId` are connected with bidirectional edges weighted at `1.0` by default. This overwrites the `ind.twitter.*` files from Step 2.

---

### Step 6 — K-fold CV: with conversation edges

Same command as Step 4, but now the graph on disk contains conversation edges. The same pretrained BERT checkpoint is reused.

```bash
python run_kfold_twitter.py \
    --k 5 \
    --seed 42 \
    --m 0.7 \
    --nb_epochs 5 \
    --device cuda \
    --bert_init jcblaise/roberta-tagalog-base \
    --pretrained_bert_ckpt ./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth \
    --summary_file kfold_with_conv_edges_m0.7_summary.csv
```

---

## Running the Full Pipeline Automatically

A single shell script runs all steps sequentially:

```bash
chmod +x run_twitter_ablation.sh
./run_twitter_ablation.sh
```

Configurable parameters are defined at the top of the script:

```bash
SEED=42
K=5
NB_EPOCHS=5
DEVICE=cuda
BERT_INIT="jcblaise/roberta-tagalog-base"
BERT_LR=2e-5
BERT_FINETUNE_EPOCHS=10
TEST_SPLIT_RATIO=0.1
M_VALUE=0.7
```

The script will exit immediately if any step fails (`set -e`). It also validates that the BERT checkpoint was saved before proceeding to the k-fold runs.

---

## K-Fold Split

With `k=5`, each fold gets:
- **Test**: 20% of all documents (1 fold out of 5)
- **Remaining**: 80% goes to the fold's training pool

Then inside `train_bert_gcn.py`, the 80% training pool is split 90/10 via `StratifiedShuffleSplit`:
- **Real train**: 72% of all documents (90% of 80%)
- **Val**: 8% of all documents (10% of 80%)

So each fold's effective split is:
```
Real train : 72%
Val        :  8%
Test       : 20%
```

---

## Output Structure

### Per-fold BertGCN checkpoints

Each fold produces a timestamped directory:

```
./checkpoint/twitter_fold{fold_id}_seed{seed}_gcn_{timestamp}/
├── checkpoint.pth       — best model weights (by val accuracy)
├── final_results.txt    — test set metrics for this fold
├── training.log         — epoch-by-epoch train/val accuracy and loss
└── train_bert_gcn.py    — copy of the training script used
```

### Finetuned BERT checkpoint

```
./checkpoint/jcblaise/roberta-tagalog-base_twitter/
├── checkpoint.pth       — best BERT weights (by val accuracy)
└── training.log
```

### Summary CSVs

One CSV per condition:

```
kfold_no_conv_edges_m0.7_summary.csv
kfold_with_conv_edges_m0.7_summary.csv
```

Each CSV contains:

```
seed,fold,accuracy,f1_macro,checkpoint
```

### Per-fold `final_results.txt` example

```
FINAL TEST RESULTS
================================================================================
Dataset: twitter
Seed: 42
K-fold mode: True

Test Accuracy: 0.8234
Test F1 Macro: 0.8101

F1 Positive: 0.8412
F1 Negative: 0.7790
F1 Neutral:  0.7901

              precision    recall  f1-score   support
    positive     0.8500    0.8326    0.8412       210
    negative     0.7901    0.7681    0.7790       180
     neutral     0.8012    0.7793    0.7901       160
    accuracy                         0.8234       550
   macro avg     0.8138    0.7933    0.8034       550
weighted avg     0.8267    0.8234    0.8248       550
```

### Aggregated terminal output (per `run_kfold_twitter.py` call)

```
======================================================================
  AGGREGATED RESULTS  seed=42  (5/5 folds)
======================================================================
  accuracy                 : 0.8223 ± 0.0071  [0.8234, 0.8102, 0.8312, 0.8198, 0.8267]
  f1_macro                 : 0.8081 ± 0.0089  [0.8101, 0.7934, 0.8201, 0.8045, 0.8123]
======================================================================
```

---

## Troubleshooting

**GPU not compatible**
```bash
# Change in run_twitter_ablation.sh
DEVICE=cpu
# Or pass directly
--device cpu
```

**Out of memory**
```bash
# Add to run_kfold_twitter.py calls
--batch_size 16
```

**BERT checkpoint not found before k-fold runs**

This means `finetune_bert.py` did not complete successfully, or the checkpoint path has changed. Verify the file exists:
```bash
ls ./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth
```

**Graph not found error**

`run_kfold_twitter.py` checks for `data/ind.twitter.adj` at startup. If missing, run the appropriate `build_graph.py` step first.

**`corpus/twitter_shuffle.txt` not found during finetuning**

`finetune_bert.py` requires this file, which is generated by `build_graph.py`. Make sure Step 2 completed successfully before running Step 3.

---

## Key Argument Reference

### `run_kfold_twitter.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--k` | `5` | Number of folds |
| `--seed` | — | Single seed (shorthand for `--seeds`) |
| `--seeds` | `[42]` | List of seeds |
| `--m` | `0.7` | BERT/GCN balance factor (fixed at 0.7) |
| `--nb_epochs` | `50` | Max training epochs per fold |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--bert_init` | `jcblaise/roberta-tagalog-base` | Base BERT model |
| `--pretrained_bert_ckpt` | `None` | Path to finetuned BERT checkpoint |
| `--keep_conversations` | `True` | Use StratifiedGroupKFold (conversation-aware splits) |
| `--no_keep_conversations` | — | Use plain StratifiedKFold instead |
| `--summary_file` | `kfold_twitter_summary.csv` | Output CSV path |

### `prepare_twt_dataset.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | — | Path to labeled tweets CSV file (required) |
| `--output_dir` | `data` | Output directory |
| `--dataset_name` | `twitter` | Dataset name |
| `--seed` | `42` | Random seed for reproducibility |
| `--test_split_ratio` | `0.2` | Ratio of data to use as test set (use `0.1` for this pipeline) |

### `build_graph.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `dataset` | — | Dataset name (positional, use `twitter`) |
| `--seed` | `42` | Random seed for shuffle |
| `--no_conversation_edges` | `False` | Omit conversation-based edges |
| `--conversation_weight` | `1.0` | Edge weight for conversation edges |

### `finetune_bert.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `20ng` | Use `twitter` |
| `--bert_init` | `roberta-base` | Base BERT model |
| `--bert_lr` | `1e-4` | Learning rate |
| `--nb_epochs` | `60` | Number of epochs |
| `--batch_size` | `128` | Batch size |