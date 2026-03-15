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
import pickle
import json

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
    unlabeled_csv_path=None,
    seed=42
):
    """
    Prepare Twitter dataset for BertGCN
    
    Args:
        csv_path: Path to labeled tweets CSV (ALL will be used as training data)
        output_dir: Output directory
        dataset_name: Name for the dataset
        unlabeled_csv_path: Optional path to unlabeled tweets CSV (becomes test set)
        seed: Random seed (for reproducibility)
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
    
    # ALL labeled data becomes 'train' in the .txt file
    df_labeled['split'] = 'train'
    
    dfs_to_combine = [df_labeled]
    
    if df_unlabeled is not None:
        df_unlabeled['split'] = 'test'
        df_unlabeled['sentiment'] = 'unlabeled'  # Placeholder
        dfs_to_combine.append(df_unlabeled)
    
    df_all = pd.concat(dfs_to_combine, ignore_index=True)
    
    print(f"\nTotal documents: {len(df_all)}")
    print(f"  Train (labeled): {len(df_labeled)}")
    print(f"  Test (unlabeled): {len(df_unlabeled) if df_unlabeled is not None else 0}")
    print(f"\nNOTE: Graph builder will create 90/10 validation split from training data")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/corpus', exist_ok=True)
    
    # Get unique sentiment labels (excluding 'unlabeled')
    sentiment_labels = [l for l in df_labeled['sentiment'].unique() if l != 'unlabeled']
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
    
    with open(f'{output_dir}/{dataset_name}_conversations.pkl', 'wb') as f:
        pickle.dump(conversation_map, f)
    
    print(f"  Total conversations: {len(conversation_map)}")
    print(f"  Conversations with multiple tweets: {sum(1 for docs in conversation_map.values() if len(docs) > 1)}")
    
    # # Create metadata file with statistics
    # metadata = {
    #     'dataset_name': dataset_name,
    #     'total_documents': int(len(df_all)),
    #     'train_size': int(len(df_labeled)),
    #     'test_size': int(len(df_unlabeled) if df_unlabeled is not None else 0),
    #     'sentiment_labels': sentiment_labels,
    #     'num_conversations': int(len(conversation_map)),
    #     'seed': int(seed),
    #     'note': 'Graph builder creates 90/10 validation split from training data internally'
    # }
    
    # with open(f'{output_dir}/{dataset_name}_metadata.json', 'w') as f:
    #     json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Build graph: python3 build_graph.py {dataset_name} --seed {seed} --conversation_weight 1.0")
    print(f"  2. Train model: python3 train_bert_gcn.py --dataset {dataset_name} --seed {seed}")
    
    # return metadata


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
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    prepare_twitter_dataset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        unlabeled_csv_path=args.unlabeled_csv,
        seed=args.seed
    )