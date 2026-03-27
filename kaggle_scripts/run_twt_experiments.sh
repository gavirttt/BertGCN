%%writefile run_twitter_experiments.sh
#!/bin/bash
# =============================================================================
# run_twitter_ablation.sh
#
# Full sequential pipeline for the Twitter BertGCN ablation study.
# Compares:
#   - Condition A: No conversation edges  (values of m from arguments)
#   - Condition B: With conversation edges (values of m from arguments)
# Both conditions use the same pretrained BERT checkpoint.
#
# Usage:
#   chmod +x run_twitter_experiments.sh
#   ./run_twitter_experiments.sh -m 0.3 0.5 0.7  # For both conditions
#   ./run_twitter_experiments.sh --m-values 0.3 0.5 0.7 --m-values-conv 0.4 0.6 0.8  # Different values for each condition
#
# Outputs:
#   ./checkpoint/twitter_fold*_seed*_*/   — per-fold BertGCN checkpoints
#   ./checkpoint/jcblaise/roberta-tagalog-base_twitter/  — finetuned BERT
#   kfold_no_conv_edges_summary_m*.csv    — results for condition A
#   kfold_with_conv_edges_summary_m*.csv  — results for condition B
# =============================================================================

set -e  # Exit immediately on any error

# ── Configurable parameters ──────────────────────────────────────────────────
SEED=42
K=5
NB_EPOCHS=5
DEVICE=cuda
BERT_INIT="jcblaise/roberta-tagalog-base"
BERT_LR=2e-5
LABELED_CSV="data/tweets_labeled_set.csv"
UNLABELED_CSV="data/tweets_unlabeled_set.csv"
M_VALUES_NO_CONV=()
M_VALUES_WITH_CONV=()

# ── Parse command line arguments ─────────────────────────────────────────────
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -m, --m-values VALUES       M values for both conditions (space-separated)"
    echo "  --m-values-no-conv VALUES   M values for condition without conversation edges"
    echo "  --m-values-conv VALUES       M values for condition with conversation edges"
    echo "  -h, --help                    Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m 0.3 0.5 0.7                    # Use same m values for both conditions"
    echo "  $0 --m-values-no-conv 0.3 0.5 --m-values-conv 0.4 0.6 0.8  # Different values"
    echo "  $0 -m 0.5 --seed 123 --k 10           # Custom seed and folds"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--m-values)
            shift
            M_VALUES_NO_CONV=()
            M_VALUES_WITH_CONV=()
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                M_VALUES_NO_CONV+=("$1")
                M_VALUES_WITH_CONV+=("$1")
                shift
            done
            ;;
        --m-values-no-conv)
            shift
            M_VALUES_NO_CONV=()
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                M_VALUES_NO_CONV+=("$1")
                shift
            done
            ;;
        --m-values-conv)
            shift
            M_VALUES_WITH_CONV=()
            while [[ $# -gt 0 && ! $1 =~ ^- ]]; do
                M_VALUES_WITH_CONV+=("$1")
                shift
            done
            ;;
        --bert-ckpt)
            shift
            BERT_CKPT="$1"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

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
echo "  BERT checkpoint : $BERT_CKPT"
echo ""
echo "  M values (no conversation edges): ${M_VALUES_NO_CONV[*]}"
echo "  M values (with conversation edges): ${M_VALUES_WITH_CONV[*]}"
echo ""

# Check if m values are provided
if [ ${#M_VALUES_NO_CONV[@]} -eq 0 ] && [ ${#M_VALUES_WITH_CONV[@]} -eq 0 ]; then
    echo "✗ ERROR: No m values provided. Use -m or --m-values to specify values."
    show_help
    exit 1
fi

# =============================================================================
# OPTION A: Build graph No conversation edges
# =============================================================================
if [ ${#M_VALUES_NO_CONV[@]} -gt 0 ]; then
    section "STEP 5 — Build Graph (No Conversation Edges)"
    
    step "Running build_graph.py (No conversation edges)"
    /kaggle/working/venv/bin/python build_graph.py twitter --seed $SEED --no_conversation_edges
    
    echo "✓ Graph built (No conversation edges)"
    
    # =============================================================================
    # K-fold CV: No conversation edges, sweep m
    # =============================================================================
    section "STEP 6 — K-Fold CV: No Conversation Edges"
    
    for M in "${M_VALUES_NO_CONV[@]}"; do
        step "m=$M | no conversation edges"
        
        # Create summary filename with m value
        SUMMARY_FILE="kfold_no_conv_edges_m${M}_summary.csv"
        
        /kaggle/working/venv/bin/python run_kfold_twitter.py \
            --k $K \
            --seed $SEED \
            --m $M \
            --nb_epochs $NB_EPOCHS \
            --device $DEVICE \
            --bert_init "$BERT_INIT" \
            --no_keep_conversations \
            --summary_file "$SUMMARY_FILE"
        
        echo "✓ Completed m=$M (no conversation edges) -> $SUMMARY_FILE"
    done
    
    echo ""
    echo "✓ All no-conversation-edges runs complete"
fi

# =============================================================================
# OPTION B: Build graph WITH conversation edges (only if needed)
# =============================================================================
if [ ${#M_VALUES_WITH_CONV[@]} -gt 0 ]; then
    section "Build Graph (With Conversation Edges)"
    
    step "Running build_graph.py (with conversation edges)"
    /kaggle/working/venv/bin/python build_graph.py twitter --seed $SEED
    
    echo "✓ Graph built (with conversation edges)"
    
    # =============================================================================
    # K-fold CV: With conversation edges, sweep m
    # =============================================================================
    section "K-Fold CV: With Conversation Edges"
    
    for M in "${M_VALUES_WITH_CONV[@]}"; do
        step "m=$M | with conversation edges"
        
        # Create summary filename with m value
        SUMMARY_FILE="kfold_with_conv_edges_m${M}_summary.csv"
        
        /kaggle/working/venv/bin/python run_kfold_twitter.py \
            --k $K \
            --seed $SEED \
            --m $M \
            --nb_epochs $NB_EPOCHS \
            --device $DEVICE \
            --bert_init "$BERT_INIT" \
            --summary_file "$SUMMARY_FILE"
        
        echo "✓ Completed m=$M (with conversation edges) -> $SUMMARY_FILE"
    done
    
    echo ""
    echo "✓ All with-conversation-edges runs complete"
fi

# =============================================================================
# DONE
# =============================================================================
section "Pipeline Complete"

echo "  Results:"
echo ""
if [ ${#M_VALUES_NO_CONV[@]} -gt 0 ]; then
    echo "  No conversation edges:"
    for M in "${M_VALUES_NO_CONV[@]}"; do
        echo "    m=$M → kfold_no_conv_edges_m${M}_summary.csv"
    done
fi
echo ""
if [ ${#M_VALUES_WITH_CONV[@]} -gt 0 ]; then
    echo "  With conversation edges:"
    for M in "${M_VALUES_WITH_CONV[@]}"; do
        echo "    m=$M → kfold_with_conv_edges_m${M}_summary.csv"
    done
fi
echo ""
echo "  BertGCN checkpoints : ./checkpoint/twitter_fold*_seed*/"
echo "  Finetuned BERT      : $BERT_CKPT"
echo ""