#!/bin/bash
# =============================================================================
# run_topic_modeling.sh
#
# Runs topic modeling on:
#   1. The whole dataset (all months, all sentiments)
#   2. Each month from December 2024 to May 2025
#
# Usage:
#   chmod +x run_topic_modeling.sh
#   ./run_topic_modeling.sh
#
# Outputs saved to: topic_modeling_results/
# =============================================================================

set -e

# ── Configurable parameters ──────────────────────────────────────────────────
CSV="data/tweets_predictions.csv"
N_TOPICS=5
DEVICE="cuda"
BERT_INIT="dost-asti/RoBERTa-tl-sentiment-analysis"
OUTPUT_DIR="topic_modeling_results"
PYTHON="/kaggle/working/venv/bin/python"

MONTHS=(
    "2024-12"
    "2025-01"
    "2025-02"
    "2025-03"
    "2025-04"
    "2025-05"
)

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Start ─────────────────────────────────────────────────────────────────────
section "Topic Modeling Pipeline"
echo "  CSV         : $CSV"
echo "  N topics    : $N_TOPICS"
echo "  Device      : $DEVICE"
echo "  BERT model  : $BERT_INIT"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Months      : ${MONTHS[*]}"
echo ""

# Validate CSV exists
if [ ! -f "$CSV" ]; then
    echo "✗ ERROR: CSV not found at $CSV"
    exit 1
fi
echo "✓ CSV found"

# =============================================================================
# RUN 1: Whole dataset (all months, all sentiments)
# =============================================================================
section "RUN 1 — Whole Dataset (all months, all sentiments)"

step "Running topic modeling on full dataset"
$PYTHON topic_modeling.py \
    --csv "$CSV" \
    --bert_init "$BERT_INIT" \
    --n_topics $N_TOPICS \
    --device $DEVICE \
    --output_dir "$OUTPUT_DIR"

echo "✓ Full dataset run complete"

# =============================================================================
# RUN 2: Per month
# =============================================================================
section "RUN 2 — Per Month"

for MONTH in "${MONTHS[@]}"; do
    step "Month: $MONTH"

    $PYTHON topic_modeling.py \
        --csv "$CSV" \
        --bert_init "$BERT_INIT" \
        --n_topics $N_TOPICS \
        --month "$MONTH" \
        --device $DEVICE \
        --output_dir "$OUTPUT_DIR"

    echo "✓ Month $MONTH complete"
done

# =============================================================================
# DONE
# =============================================================================
section "Pipeline Complete"

echo "  Outputs:"
echo ""
echo "  Full dataset:"
echo "    $OUTPUT_DIR/umap_all_all_months.png"
echo "    $OUTPUT_DIR/wordclouds_combined_all_all_months.png"
echo "    $OUTPUT_DIR/llr_keywords_all_all_months.json"
echo "    $OUTPUT_DIR/tweets_with_clusters_all_all_months.csv"
echo "    $OUTPUT_DIR/monthly_sentiment_trend.png"
echo "    $OUTPUT_DIR/monthly_sentiment_counts.csv"
echo ""
echo "  Per month:"
for MONTH in "${MONTHS[@]}"; do
    echo "    $OUTPUT_DIR/umap_all_${MONTH}.png"
    echo "    $OUTPUT_DIR/wordclouds_combined_all_${MONTH}.png"
    echo "    $OUTPUT_DIR/llr_keywords_all_${MONTH}.json"
    echo "    $OUTPUT_DIR/tweets_with_clusters_all_${MONTH}.csv"
done
echo ""