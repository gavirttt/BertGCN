"""
merge_datasets.py

Merges the labeled CSV and predictions CSV into a single unified CSV
for use with topic_modeling.py.

Labeled CSV columns:
    pseudo_id, text, retweetCount, replyCount, likeCount, quoteCount,
    viewCount, bookmarkCount, createdAt, lang, isReply,
    pseudo_conversationId, pseudo_inReplyToUsername, pseudo_author_userName,
    author_isBlueVerified, sentiment (int: 0=positive, 1=negative, 2=neutral)

Predictions CSV columns:
    pseudo_id, text, retweetCount, replyCount, likeCount, quoteCount,
    viewCount, bookmarkCount, createdAt, lang, isReply,
    pseudo_conversationId, pseudo_inReplyToUsername, pseudo_author_userName,
    author_isBlueVerified, cleaned_text, sentiment (str), prob_positive,
    prob_negative, prob_neutral

Output CSV columns (unified):
    pseudo_id, text, cleaned_text, retweetCount, replyCount, likeCount,
    quoteCount, viewCount, bookmarkCount, createdAt, lang, isReply,
    pseudo_conversationId, pseudo_inReplyToUsername, pseudo_author_userName,
    author_isBlueVerified, sentiment (str), source,
    prob_positive, prob_negative, prob_neutral (NaN for labeled rows)

    text         -> raw text, used for BERT embeddings
    cleaned_text -> cleaned text, used for LDA / word clouds

Usage:
    python merge.py --labeled data/tweets_labeled_set.csv --predictions data/tweets_predictions.csv --output data/tweets_labeled_full.csv
"""

import argparse
import pandas as pd
import sys
sys.path.append('.')
from prepare_twt_dataset import clean_text

parser = argparse.ArgumentParser()
parser.add_argument('--labeled',      type=str, required=True,
                    help='Path to manually labeled CSV')
parser.add_argument('--predictions',  type=str, required=True,
                    help='Path to model predictions CSV')
parser.add_argument('--output',       type=str, default='data/tweets_combined.csv',
                    help='Output path for merged CSV')
args = parser.parse_args()

label_map_int = {0: 'positive', 1: 'negative', 2: 'neutral'}

# ── Load labeled CSV ──────────────────────────────────────────────────────────
print(f"Loading labeled CSV: {args.labeled}")
df_labeled = pd.read_csv(args.labeled)
print(f"  Rows    : {len(df_labeled)}")
print(f"  Columns : {df_labeled.columns.tolist()}")

# Normalize integer sentiment to string
df_labeled['sentiment'] = df_labeled['sentiment'].map(label_map_int)
df_labeled['source'] = 'labeled'

# Clean text for labeled rows — produces cleaned_text column to match predictions CSV
print("  Cleaning text for labeled rows...")
df_labeled['cleaned_text'] = df_labeled['text'].apply(clean_text)

# ── Load predictions CSV ──────────────────────────────────────────────────────
print(f"\nLoading predictions CSV: {args.predictions}")
df_pred = pd.read_csv(args.predictions)
print(f"  Rows    : {len(df_pred)}")
print(f"  Columns : {df_pred.columns.tolist()}")

# Normalize string sentiment just in case
df_pred['sentiment'] = df_pred['sentiment'].str.lower().str.strip()
df_pred['source'] = 'predicted'

# ── Check for duplicates between the two CSVs ─────────────────────────────────
labeled_ids = set(df_labeled['pseudo_id'].astype(str))
pred_ids    = set(df_pred['pseudo_id'].astype(str))
overlap     = labeled_ids & pred_ids

if overlap:
    print(f"\n WARNING: {len(overlap)} pseudo_ids appear in both CSVs.")
    print(f"  Keeping labeled version for overlapping IDs (ground truth preferred).")
    df_pred = df_pred[~df_pred['pseudo_id'].astype(str).isin(overlap)].reset_index(drop=True)
    print(f"  Predictions after dedup: {len(df_pred)}")
else:
    print(f"\n No overlapping pseudo_ids between datasets.")

# ── Define shared columns ─────────────────────────────────────────────────────
# Both dataframes now have text + cleaned_text
# text         -> raw text, for BERT embeddings
# cleaned_text -> cleaned text, for LDA / word clouds
shared_cols = [
    'pseudo_id', 'text', 'cleaned_text', 'retweetCount', 'replyCount',
    'likeCount', 'quoteCount', 'viewCount', 'bookmarkCount', 'createdAt',
    'lang', 'isReply', 'pseudo_conversationId', 'pseudo_inReplyToUsername',
    'pseudo_author_userName', 'author_isBlueVerified', 'sentiment', 'source'
]

# Optional columns only in predictions CSV (NaN for labeled rows after concat)
optional_cols = ['prob_positive', 'prob_negative', 'prob_neutral']

# Keep only columns that exist in each dataframe
labeled_cols = [c for c in shared_cols if c in df_labeled.columns]
pred_cols    = [c for c in shared_cols + optional_cols if c in df_pred.columns]

df_labeled = df_labeled[labeled_cols]
df_pred    = df_pred[pred_cols]

# ── Merge ─────────────────────────────────────────────────────────────────────
print("\nMerging datasets...")
df_combined = pd.concat([df_labeled, df_pred], ignore_index=True)

# ── Validate ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"MERGED DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total rows       : {len(df_combined)}")
print(f"  From labeled   : {len(df_labeled)}")
print(f"  From predicted : {len(df_pred)}")
print(f"\nSentiment distribution:")
print(df_combined['sentiment'].value_counts().to_string())
print(f"\nSource distribution:")
print(df_combined['source'].value_counts().to_string())
print(f"\nCleaned text coverage:")
print(f"  Rows with cleaned_text : {df_combined['cleaned_text'].notna().sum()}")
print(f"  Rows missing           : {df_combined['cleaned_text'].isna().sum()}")

# Parse dates and show monthly distribution
df_combined['createdAt'] = pd.to_datetime(
    df_combined['createdAt'], infer_datetime_format=True)
df_combined['year_month'] = df_combined['createdAt'].dt.to_period('M')
print(f"\nDate range: {df_combined['year_month'].min()} to {df_combined['year_month'].max()}")
print(f"\nMonthly tweet counts:")
print(df_combined['year_month'].value_counts().sort_index().to_string())

print(f"\nMissing values:")
missing = df_combined.isnull().sum()
missing = missing[missing > 0]
print(missing.to_string() if len(missing) > 0 else "  None")

# ── Save ──────────────────────────────────────────────────────────────────────
df_combined.drop(columns=['year_month'], inplace=True)  # drop temp column
df_combined.to_csv(args.output, index=False)
print(f"\n Merged CSV saved to: {args.output}")
print(f"  Final columns: {df_combined.columns.tolist()}")