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

    # Convert emojis to plain English word tokens
    def replace_emoji_handler(emoji_char, data_copy):
        return ' ' + emoji.demojize(emoji_char).replace(':', '').replace('_', ' ') + ' '
    text = emoji.replace_emoji(text, replace=replace_emoji_handler)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Normalize quotation marks and apostrophes
    text = text.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')

    # Preserve acronyms: temporarily replace periods between single letters with a placeholder
    # e.g. U.S.A → U<DOT>S<DOT>A
    text = re.sub(r'(?<=[A-Za-z])\.(?=[A-Za-z]\.)', '<DOT>', text)  # middle dots in acronym
    text = re.sub(r'(?<=[A-Za-z]{1})\.(?=[A-Za-z]{1}\b)', '<DOT>', text)  # trailing dot in acronym

    # Collapse repeated punctuation
    text = re.sub(r'([!?.,])\1+', r'\1', text)

    # Pad punctuation with spaces so they detach from words
    text = re.sub(r'([.,!?;:"\(\)\[\]{}])', r' \1 ', text)

    # Remove acronym dots
    text = text.replace('<DOT>', '')

    # Normalize mentions and hashtags
    text = re.sub(r'@{2,}', '@', text)
    text = re.sub(r'@\s+', '@', text)
    text = re.sub(r'#{2,}', '#', text)
    text = re.sub(r'#\s+', '#', text)

    # Remove non-ASCII characters
    text = text.encode('ascii', errors='ignore').decode()

    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def prepare_twitter_dataset(
    csv_path,
    output_dir='data',
    dataset_name='twitter',
    seed=42,
    test_split_ratio=0.2
):
    """
    Prepare Twitter dataset for BertGCN
    
    Args:
        csv_path: Path to labeled tweets CSV
        output_dir: Output directory
        dataset_name: Name for the dataset
        seed: Random seed (for reproducibility)
        test_split_ratio: Ratio of data to use as test set (default: 0.2)
    """
    
    print(f"Loading labeled dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['pseudo_id', 'text', 'sentiment', 'pseudo_conversationId']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Remove rows with missing text or sentiment
    df = df.dropna(subset=['text', 'sentiment'])
    
    print(f"Total labeled tweets: {len(df)}")
    print(f"Sentiment distribution:\n{df['sentiment'].value_counts()}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split into train and test
    test_size = int(len(df) * test_split_ratio)
    df_test = df[:test_size].copy()
    df_train = df[test_size:].copy()
    
    print(f"\nSplit results:")
    print(f"  Train samples: {len(df_train)} ({test_split_ratio*100:.0f}% of total)")
    print(f"  Test samples: {len(df_test)} ({(1-test_split_ratio)*100:.0f}% of total)")
    
    # Assign splits for the .txt file
    df_train['split'] = 'train'
    df_test['split'] = 'test'
    
    # Combine for processing
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    print(f"\nNOTE: Graph builder will create 90/10 validation split from training data internally")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/corpus', exist_ok=True)
    
    # Get unique sentiment labels
    sentiment_labels = sorted(df['sentiment'].unique())
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
    
    # Create metadata file with statistics
    metadata = {
        'dataset_name': dataset_name,
        'total_documents': int(len(df_all)),
        'train_size': int(len(df_train)),
        'test_size': int(len(df_test)),
        'test_split_ratio': test_split_ratio,
        'sentiment_labels': sentiment_labels,
        'num_conversations': int(len(conversation_map)),
        'seed': int(seed),
        'note': 'Graph builder creates 90/10 validation split from training data internally'
    }
    
    with open(f'{output_dir}/{dataset_name}_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"  1. Build graph: python3 build_graph.py {dataset_name} --seed {seed} --conversation_weight 1.0")
    print(f"  2. Train model: python3 train_bert_gcn.py --dataset {dataset_name} --seed {seed}")
    print(f"  3. Or run k-fold CV: python3 run_kfold_twitter.py --seeds {seed} --k 5")
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare Twitter dataset for BertGCN')
    parser.add_argument('--csv', type=str, required=True,
                       help='Path to labeled tweets CSV file')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--dataset_name', type=str, default='twitter',
                       help='Dataset name (default: twitter)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--test_split_ratio', type=float, default=0.2,
                       help='Ratio of data to use as test set (default: 0.2)')
    
    args = parser.parse_args()
    
    prepare_twitter_dataset(
        csv_path=args.csv,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        seed=args.seed,
        test_split_ratio=args.test_split_ratio
    )