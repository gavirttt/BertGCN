"""
Script to prepare custom Twitter dataset for BertGCN
Supports CSV format with conversation connections
"""
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import re, html, emoji


def clean_text(text):
    """Clean tweet text"""
    if pd.isna(text):
        return ""
    
    # Decode HTML entities
    text = html.unescape(str(text))

    # Extract and replace emojis with single tokens
    emoji_list = []
    def replace_emoji_handler(emoji_char, data_copy):
        emoji_name = emoji.demojize(emoji_char).replace(':', '').replace('_', ' ')
        token = f"[EMOJI_{emoji_name}]"
        emoji_list.append(token)
        return token
    
    # Find all emojis and replace with single tokens
    text = emoji.replace_emoji(text, replace=replace_emoji_handler)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Normalize quotation marks and apostrophes
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    
    # Remove unwanted symbols but KEEP @ and #
    text = re.sub(r"[^a-zA-Z0-9\s\[\]'\".,!?@#-]", " ", text)
    
    # Collapse repeated punctuation
    text = re.sub(r"([!?.,])\1+", r"\1", text)

    # Normalize mentions and hashtags
    text = re.sub(r"@{2,}", "@", text)
    text = re.sub(r"@\s+", "@", text)
    text = re.sub(r"#{2,}", "#", text)
    text = re.sub(r"#\s+", "#", text)

    # Fix spacing around quotes
    text = re.sub(r'\s*"\s*', '"', text)
    text = re.sub(r"\s*'\s*", "'", text)

    # Remove non-ASCII characters
    text = text.encode("ascii", errors="ignore").decode()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def prepare_twitter_dataset(
    csv_path,
    output_dir='data',
    dataset_name='twitter',
    train_ratio=0.8,
    val_ratio=0.2,
    unlabeled_csv_path=None,
    seed=42
):
    """
    Prepare Twitter dataset for BertGCN
    
    Args:
        csv_path: Path to labeled tweets CSV
        output_dir: Output directory
        dataset_name: Name for the dataset
        train_ratio: Ratio of labeled data for training (default: 0.8)
        val_ratio: Ratio of labeled data for validation (default: 0.2)
        unlabeled_csv_path: Optional path to unlabeled tweets CSV (will be used as test set)
        seed: Random seed
    
    Note:
        - Train/Val splits come ONLY from labeled data
        - Test set is ONLY unlabeled data (if provided)
        - If no unlabeled data, a warning is shown
    """
    
    print(f"Loading labeled dataset from {csv_path}...")
    df_labeled = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['pseudo_id', 'text', 'sentiment', 'pseudo_conversationId']
    missing_cols = [col for col in required_cols if col not in df_labeled.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing text or sentiment
    df_labeled = df_labeled.dropna(subset=['text', 'sentiment'])
    
    print(f"Labeled tweets: {len(df_labeled)}")
    print(f"Sentiment distribution:\n{df_labeled['sentiment'].value_counts()}")
    
    # Load unlabeled data if provided
    df_unlabeled = None
    if unlabeled_csv_path and os.path.exists(unlabeled_csv_path):
        print(f"\nLoading unlabeled dataset from {unlabeled_csv_path}...")
        df_unlabeled = pd.read_csv(unlabeled_csv_path)
        df_unlabeled = df_unlabeled.dropna(subset=['text'])
        print(f"Unlabeled tweets (will be test set): {len(df_unlabeled)}")
    else:
        print(f"\n⚠ WARNING: No unlabeled data provided!")
        print(f"  The test set will be empty.")
        print(f"  This is only for graph building - you won't be able to evaluate on unlabeled data.")
    
    # Set random seed
    np.random.seed(seed)
    
    # Split labeled data into train/val ONLY
    df_labeled = df_labeled.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n_labeled = len(df_labeled)
    n_train = int(n_labeled * train_ratio)
    
    df_train = df_labeled[:n_train].copy()
    df_val = df_labeled[n_train:].copy()
    
    print(f"\nSplit labeled data:")
    print(f"  Train: {len(df_train)} ({train_ratio*100:.0f}% of labeled)")
    print(f"  Val: {len(df_val)} ({val_ratio*100:.0f}% of labeled)")
    
    # Combine all data
    df_train['split'] = 'train'
    df_val['split'] = 'val'
    
    dfs_to_combine = [df_train, df_val]
    
    # Unlabeled data becomes the test set
    if df_unlabeled is not None:
        df_unlabeled['split'] = 'test'
        df_unlabeled['sentiment'] = 'unlabeled'  # Placeholder
        dfs_to_combine.append(df_unlabeled)
        print(f"  Test (unlabeled): {len(df_unlabeled)}")
    
    df_all = pd.concat(dfs_to_combine, ignore_index=True)
    
    print(f"\nTotal documents: {len(df_all)}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/corpus', exist_ok=True)
    
    # Get unique sentiment labels (excluding 'unlabeled')
    sentiment_labels = df_labeled['sentiment'].unique().tolist()
    sentiment_labels.sort()
    
    print(f"\nSentiment labels: {sentiment_labels}")
    
    # Create document list file (doc_id, split, label)
    print("\nCreating document list...")
    with open(f'{output_dir}/{dataset_name}.txt', 'w', encoding='utf-8') as f:
        for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Writing docs"):
            doc_id = f"doc_{idx}"
            split = row['split']
            label = row['sentiment']
            f.write(f"{doc_id}\t{split}\t{label}\n")
    
    # Create corpus files (raw and cleaned text)
    print("\nCreating corpus files...")
    with open(f'{output_dir}/corpus/{dataset_name}.txt', 'w', encoding='utf-8') as f_raw, \
         open(f'{output_dir}/corpus/{dataset_name}.clean.txt', 'w', encoding='utf-8') as f_clean:
        
        for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Processing text"):
            text = str(row['text'])
            cleaned = clean_text(text)
            
            # Write raw text (with basic cleaning)
            f_raw.write(text.replace('\n', ' ').replace('\t', ' ') + '\n')
            
            # Write cleaned text
            f_clean.write(cleaned + '\n')
    
    # Create conversation mapping file for graph building
    print("\nCreating conversation mapping...")
    conversation_map = {}
    
    for idx, row in tqdm(df_all.iterrows(), total=len(df_all), desc="Mapping conversations"):
        conv_id = row['pseudo_conversationId']
        if pd.notna(conv_id):
            if conv_id not in conversation_map:
                conversation_map[conv_id] = []
            conversation_map[conv_id].append(idx)
    
    # Save conversation mapping
    import pickle
    with open(f'{output_dir}/{dataset_name}_conversations.pkl', 'wb') as f:
        pickle.dump(conversation_map, f)
    
    print(f"  Total conversations: {len(conversation_map)}")
    print(f"  Conversations with multiple tweets: {sum(1 for docs in conversation_map.values() if len(docs) > 1)}")
    
    # Create metadata file with statistics
    metadata = {
        'dataset_name': dataset_name,
        'total_documents': len(df_all),
        'train_size': len(df_train),
        'val_size': len(df_val),
        'test_size': len(df_unlabeled) if df_unlabeled is not None else 0,
        'sentiment_labels': sentiment_labels,
        'num_conversations': len(conversation_map),
        'seed': seed,
        'note': 'Train/Val from labeled data only. Test is unlabeled data.'
    }
    
    import json
    with open(f'{output_dir}/{dataset_name}_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"  Files saved to: {output_dir}/")
    print(f"\nDataset structure:")
    print(f"  Train: {len(df_train)} labeled tweets")
    print(f"  Val: {len(df_val)} labeled tweets")
    print(f"  Test: {len(df_unlabeled) if df_unlabeled is not None else 0} unlabeled tweets")
    print(f"\nNext steps:")
    print(f"  1. Build graph: python3 build_graph_twitter.py {dataset_name} --seed {seed}")
    print(f"  2. Train model: python3 train_bert_gcn.py --dataset {dataset_name} --seed {seed}")
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Twitter dataset for BertGCN')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to labeled tweets CSV file')
    parser.add_argument('--unlabeled_csv', type=str, default=None,
                       help='Path to unlabeled tweets CSV file (will become test set)')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--dataset_name', type=str, default='twitter',
                       help='Dataset name (default: twitter)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Train ratio from labeled data (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                       help='Validation ratio from labeled data (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    if abs(args.train_ratio + args.val_ratio - 1.0) > 0.01:
        raise ValueError("Train and val ratios must sum to 1.0")
    
    prepare_twitter_dataset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        unlabeled_csv_path=args.unlabeled_csv,
        seed=args.seed
    )