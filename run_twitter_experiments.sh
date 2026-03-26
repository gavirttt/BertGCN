#!/bin/bash
# =============================================================================
# run_twitter_ablation.sh
#
# Full sequential pipeline for the Twitter BertGCN ablation study.
# Compares:
#   - Condition A: No conversation edges  (4 values of m)
#   - Condition B: With conversation edges (4 values of m)
# Both conditions use the same pretrained BERT checkpoint.
#
# Usage:
#   chmod +x run_twitter_ablation.sh
#   ./run_twitter_ablation.sh
#
# Outputs:
#   ./checkpoint/twitter_fold*_seed*_*/   — per-fold BertGCN checkpoints
#   ./checkpoint/jcblaise/roberta-tagalog-base_twitter/  — finetuned BERT
#   kfold_no_conv_edges_summary.csv       — results for condition A
#   kfold_with_conv_edges_summary.csv     — results for condition B
# =============================================================================

set -e  # Exit immediately on any error

# ── Configurable parameters ──────────────────────────────────────────────────
SEED=42
K=5
NB_EPOCHS=5
DEVICE=cuda
BERT_INIT="jcblaise/roberta-tagalog-base"
BERT_LR=2e-5
BERT_FINETUNE_EPOCHS=10
LABELED_CSV="data/tweets_labeled_set.csv"
UNLABELED_CSV="data/tweets_unlabeled_set.csv"
BERT_CKPT="./checkpoint/jcblaise/roberta-tagalog-base_twitter/checkpoint.pth"

# m values to sweep
M_VALUES=(0.3 0.5 0.7 1.0)

# ── Helpers ──────────────────────────────────────────────────────────────────
section() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf  "║  %-64s║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════════╝"
}

step() {
    echo ""
    echo "──────────────────────────────────────────────────────────────────"
    echo "  ▶ $1"
    echo "──────────────────────────────────────────────────────────────────"
}

# ── Start ────────────────────────────────────────────────────────────────────
section "Twitter BertGCN Ablation Pipeline"
echo "  Seed        : $SEED"
echo "  K-folds     : $K"
echo "  Epochs/fold : $NB_EPOCHS"
echo "  Device      : $DEVICE"
echo "  BERT model  : $BERT_INIT"
echo "  m values    : ${M_VALUES[*]}"
echo ""

# =============================================================================
# STEP 1 — Prepare dataset (only needs to run once)
# =============================================================================
section "STEP 1 — Prepare Twitter Dataset"

step "Running prepare_twt_dataset.py"
python prepare_twt_dataset.py \
    --csv "$LABELED_CSV"

echo "✓ Dataset prepared"

# =============================================================================
# STEP 2 — Build graph WITHOUT conversation edges
# =============================================================================
section "STEP 2 — Build Graph (No Conversation Edges)"

step "Running build_graph.py --no_conversation_edges"
python build_graph.py twitter --seed $SEED --no_conversation_edges

echo "✓ Graph built (no conversation edges)"

# =============================================================================
# STEP 3 — Finetune BERT (only needs to run once; graph structure not used)
# =============================================================================
section "STEP 3 — Finetune BERT on Twitter Dataset"

step "Running finetune_bert.py"
python finetune_bert.py \
    --dataset twitter \
    --bert_init "$BERT_INIT" \
    --bert_lr $BERT_LR \
    --nb_epochs $BERT_FINETUNE_EPOCHS

echo "✓ BERT finetuned — checkpoint at: $BERT_CKPT"

# Verify the checkpoint was actually saved
if [ ! -f "$BERT_CKPT" ]; then
    echo "✗ ERROR: BERT checkpoint not found at $BERT_CKPT"
    echo "  Check that finetune_bert.py completed successfully."
    exit 1
fi

# =============================================================================
# STEP 4 — K-fold CV: No conversation edges, sweep m
# =============================================================================
section "STEP 4 — K-Fold CV: No Conversation Edges"

for M in "${M_VALUES[@]}"; do
    step "m=$M | no conversation edges"
    python run_kfold_twitter.py \
        --k $K \
        --seed $SEED \
        --m $M \
        --nb_epochs $NB_EPOCHS \
        --device $DEVICE \
        --bert_init "$BERT_INIT" \
        --pretrained_bert_ckpt "$BERT_CKPT" \
        --no_keep_conversations \
        --summary_file "kfold_no_conv_edges_m${M}_summary.csv"
    echo "✓ Completed m=$M (no conversation edges)"
done

echo ""
echo "✓ All no-conversation-edges runs complete"

# =============================================================================
# STEP 5 — Build graph WITH conversation edges
# =============================================================================
section "STEP 5 — Build Graph (With Conversation Edges)"

step "Running build_graph.py (with conversation edges)"
python build_graph.py twitter --seed $SEED

echo "✓ Graph built (with conversation edges)"

# =============================================================================
# STEP 6 — K-fold CV: With conversation edges, sweep m
# =============================================================================
section "STEP 6 — K-Fold CV: With Conversation Edges"

for M in "${M_VALUES[@]}"; do
    step "m=$M | with conversation edges"
    python run_kfold_twitter.py \
        --k $K \
        --seed $SEED \
        --m $M \
        --nb_epochs $NB_EPOCHS \
        --device $DEVICE \
        --bert_init "$BERT_INIT" \
        --pretrained_bert_ckpt "$BERT_CKPT" \
        --summary_file "kfold_with_conv_edges_m${M}_summary.csv"
    echo "✓ Completed m=$M (with conversation edges)"
done

echo ""
echo "✓ All with-conversation-edges runs complete"

# =============================================================================
# DONE
# =============================================================================
section "Pipeline Complete"

echo "  Results:"
echo ""
echo "  No conversation edges:"
for M in "${M_VALUES[@]}"; do
    echo "    m=$M → kfold_no_conv_edges_m${M}_summary.csv"
done
echo ""
echo "  With conversation edges:"
for M in "${M_VALUES[@]}"; do
    echo "    m=$M → kfold_with_conv_edges_m${M}_summary.csv"
done
echo ""
echo "  BertGCN checkpoints : ./checkpoint/twitter_fold*_seed*/"
echo "  Finetuned BERT      : $BERT_CKPT"
echo ""