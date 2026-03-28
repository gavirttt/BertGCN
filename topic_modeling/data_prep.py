"""
Data Preparation for topic modeling
    Merges a labeled CSV and a predictions CSV into a single unified CSV.
    Adds a `cleaned_text2` column by applying an extended cleaning pipeline
    to the raw `text` column

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

Output CSV columns:
    pseudo_id, text, cleaned_text, cleaned_text2, retweetCount, replyCount,
    likeCount, quoteCount, viewCount, bookmarkCount, createdAt, lang, isReply,
    pseudo_conversationId, pseudo_inReplyToUsername, pseudo_author_userName,
    author_isBlueVerified, sentiment (str), source,
    prob_positive, prob_negative, prob_neutral (NaN for labeled rows)

    text          -> raw text, used for BERT embeddings
    cleaned_text  -> cleaned text from predictions pipeline, used for LDA / word clouds
    cleaned_text2 -> extended cleaned text (this script), resolves obfuscated mentions

Usage:
    python data_prep.py --labeled data/tweets_labeled_set.csv --predictions data/tweets_predictions.csv --authors data/well_known_authors_philippine_elections.csv --output data/tweets_labeled_full.csv
"""

import re
import html
import argparse
import sys

import pandas as pd
import emoji

sys.path.append(".")
from prepare_twt_dataset import clean_text


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Merge labeled + predictions CSVs and add cleaned_text2 column.")
parser.add_argument("--labeled",type=str,required=True,
    help="Path to manually labeled CSV.")
parser.add_argument("--predictions",type=str,required=True,
    help="Path to model predictions CSV.")
parser.add_argument("--authors",type=str,default="well_known_authors_philippine_elections.csv",
    help="Path to the author hashid lookup CSV (default: well_known_authors_philippine_elections.csv).")
parser.add_argument("--output",type=str,default="data/tweets_labeled_full.csv",
    help="Output path for the merged + cleaned CSV (default: data/tweets_labeled_full.csv).")
args = parser.parse_args()

label_map_int = {0: "positive", 1: "negative", 2: "neutral"}

# ── Load labeled CSV ──────────────────────────────────────────────────────────
print(f"[Stage 1] Loading labeled CSV: {args.labeled}")
df_labeled = pd.read_csv(args.labeled)
print(f"  Rows    : {len(df_labeled)}")
print(f"  Columns : {df_labeled.columns.tolist()}")

# Normalize integer sentiment to string
df_labeled["sentiment"] = df_labeled["sentiment"].map(label_map_int)
df_labeled["source"] = "labeled"

# Clean text for labeled rows — produces cleaned_text column to match predictions CSV
print("  Cleaning text for labeled rows...")
df_labeled["cleaned_text"] = df_labeled["text"].apply(clean_text)

# ── Load predictions CSV ──────────────────────────────────────────────────────
print(f"\n[Stage 1] Loading predictions CSV: {args.predictions}")
df_pred = pd.read_csv(args.predictions)
print(f"  Rows    : {len(df_pred)}")
print(f"  Columns : {df_pred.columns.tolist()}")

# Normalize string sentiment just in case
df_pred["sentiment"] = df_pred["sentiment"].str.lower().str.strip()
df_pred["source"] = "predicted"

# ── Deduplicate overlapping pseudo_ids ────────────────────────────────────────
labeled_ids = set(df_labeled["pseudo_id"].astype(str))
pred_ids = set(df_pred["pseudo_id"].astype(str))
overlap = labeled_ids & pred_ids

if overlap:
    print(f"\n  WARNING: {len(overlap)} pseudo_ids appear in both CSVs.")
    print("  Keeping labeled version for overlapping IDs (ground truth preferred).")
    df_pred = df_pred[
        ~df_pred["pseudo_id"].astype(str).isin(overlap)
    ].reset_index(drop=True)
    print(f"  Predictions after dedup: {len(df_pred)}")
else:
    print("\n  No overlapping pseudo_ids between datasets.")

# ── Select and align columns ──────────────────────────────────────────────────
shared_cols = [
    "pseudo_id", "text", "cleaned_text", "retweetCount", "replyCount",
    "likeCount", "quoteCount", "viewCount", "bookmarkCount", "createdAt",
    "lang", "isReply", "pseudo_conversationId", "pseudo_inReplyToUsername",
    "pseudo_author_userName", "author_isBlueVerified", "sentiment", "source",
]
optional_cols = ["prob_positive", "prob_negative", "prob_neutral"]

labeled_cols = [c for c in shared_cols if c in df_labeled.columns]
pred_cols = [c for c in shared_cols + optional_cols if c in df_pred.columns]

df_labeled = df_labeled[labeled_cols]
df_pred = df_pred[pred_cols]

# ── Concatenate ───────────────────────────────────────────────────────────────
print("\n[Stage 1] Merging datasets...")
df = pd.concat([df_labeled, df_pred], ignore_index=True)

# ── Merge summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MERGED DATASET SUMMARY")
print(f"{'='*60}")
print(f"Total rows       : {len(df)}")
print(f"  From labeled   : {len(df_labeled)}")
print(f"  From predicted : {len(df_pred)}")
print(f"\nSentiment distribution:")
print(df["sentiment"].value_counts().to_string())
print(f"\nSource distribution:")
print(df["source"].value_counts().to_string())
print(f"\nCleaned text coverage:")
print(f"  Rows with cleaned_text : {df['cleaned_text'].notna().sum()}")
print(f"  Rows missing           : {df['cleaned_text'].isna().sum()}")

df["createdAt"] = pd.to_datetime(df["createdAt"], infer_datetime_format=True)
df["year_month"] = df["createdAt"].dt.to_period("M")
print(f"\nDate range: {df['year_month'].min()} to {df['year_month'].max()}")
print("\nMonthly tweet counts:")
print(df["year_month"].value_counts().sort_index().to_string())
print("\nMissing values:")
missing = df.isnull().sum()
missing = missing[missing > 0]
print(missing.to_string() if len(missing) > 0 else "  None")

df.drop(columns=["year_month"], inplace=True)


# ── Load author hashid lookup ─────────────────────────────────────────────────
print(f"\n[Stage 2] Loading author lookup from: {args.authors}")
authors_df = pd.read_csv(args.authors)
print(f"  {len(authors_df):,} known authors loaded.")

# Build hashid → real_username mapping
hashid_to_username = {}
for _, row in authors_df.iterrows():
    hashid = str(row["obfuscated_userName"]).lstrip("@").strip()
    real = str(row["author_userName"]).lstrip("@").strip()
    if hashid and real:
        hashid_to_username[hashid] = real

print(f"  Hashid → username mappings built: {len(hashid_to_username)}")

# ── Mention resolver ──────────────────────────────────────────────────────────
_MENTION_RE = re.compile(r"@(\w+)")


def resolve_mentions(text):
    """Replace obfuscated mentions with real usernames where known."""
    def _replace(m: re.Match) -> str:
        handle = m.group(1)
        if handle in hashid_to_username:
            return "@" + hashid_to_username[handle]
        return m.group(0)  # leave unknown mentions unchanged

    return _MENTION_RE.sub(_replace, text)


# ── Core cleaning function ────────────────────────────────────────────────────
def clean_text2(text):
    """
    Clean a raw tweet text for cleaned_text2:
      - Emojis are REMOVED (not converted to text).
      - Obfuscated mentions are resolved to real usernames.
    """
    if pd.isna(text):
        return ""

    # 1. Decode HTML entities  (&amp; → &, &lt; → <, …)
    text = html.unescape(text)

    # 2. Remove emojis entirely
    text = emoji.replace_emoji(text, replace="")

    # 3. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # 4. Normalize quotation marks and apostrophes
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')

    # 5a. Preserve acronyms: U.S.A. → U<DOT>S<DOT>A
    text = re.sub(r"(?<=[A-Za-z])\.(?=[A-Za-z]\.)", "<DOT>", text)
    text = re.sub(r"(?<=[A-Za-z]{1})\.(?=[A-Za-z]{1}\b)", "<DOT>", text)

    # 5b. Collapse repeated punctuation  (!! → !, ... → .)
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # 5c. Pad punctuation with spaces so they detach from words
    text = re.sub(r'([.,!?;:"\(\)\[\]{}])', r" \1 ", text)

    # 5d. Remove acronym placeholder dots
    text = text.replace("<DOT>", "")

    # 6. Normalize mentions BEFORE stripping non-ASCII so hashids survive
    text = re.sub(r"@{2,}", "@", text)
    text = re.sub(r"@\s+", "@", text)

    # 7. Resolve obfuscated mentions → real usernames
    text = resolve_mentions(text)

    # 8. Normalize hashtags
    text = re.sub(r"#{2,}", "#", text)
    text = re.sub(r"#\s+", "#", text)

    # 9. Strip non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()

    # 10. Collapse extra whitespace
    text = " ".join(text.split())

    return text.strip()


# ── Apply ─────────────────────────────────────────────────────────────────────
print("\n[Stage 2] Generating cleaned_text2 ...")
df["cleaned_text2"] = df["text"].apply(clean_text2)

# Insert cleaned_text2 immediately after cleaned_text (if that column exists)
if "cleaned_text" in df.columns:
    ct_pos = df.columns.get_loc("cleaned_text")
    cols = df.columns.tolist()
    cols.remove("cleaned_text2")
    cols.insert(ct_pos + 1, "cleaned_text2")
    df = df[cols]

# Quick sanity-check: show a few rows
sample = df[["text", "cleaned_text2"]].head(5)
print("\nSample (first 5 rows):")
for i, row in sample.iterrows():
    print(f"\n  [original]  {row['text'][:120]}")
    print(f"  [cleaned2]  {row['cleaned_text2'][:120]}")

changed = (df["text"].fillna("") != df["cleaned_text2"].fillna("")).sum()
print(f"\nRows where text changed: {changed:,} / {len(df):,}")

# ── Save file ─────────────────────────────────────────────────────────────────
df.to_csv(args.output, index=False)
print(f"\nSaved to: {args.output}")
print(f"Final columns: {df.columns.tolist()}")
print("Done ✓")