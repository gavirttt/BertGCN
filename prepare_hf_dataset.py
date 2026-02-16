"""
Script to prepare HuggingFace datasets for BertGCN
Supports: viethq1906/isarcasm_2022_taskA_En and viethq1906/semeval_2018_3A
"""
import os
import sys
import argparse
from datasets import load_dataset
from tqdm import tqdm

# Set console encoding to UTF-8
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def prepare_isarcasm(output_dir='data'):
    """Prepare iSarcasm dataset"""
    print("Loading iSarcasm dataset from HuggingFace...")
    dataset = load_dataset('viethq1906/isarcasm_2022_taskA_En')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/corpus', exist_ok=True)
    dataset_name = 'isarcasm'
    
    # Create data files - specify utf-8 encoding
    with open(f'{output_dir}/{dataset_name}.txt', 'w', encoding='utf-8') as f:
        idx = 0
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                for item in dataset[split_name]:
                    label = 'sarcastic' if item['sentiment'] == 1 else 'not_sarcastic'
                    f.write(f"doc_{idx}\t{split_name}\t{label}\n")
                    idx += 1
    
    # Create corpus file - specify utf-8 encoding and handle potential encoding errors
    with open(f'{output_dir}/corpus/{dataset_name}.txt', 'w', encoding='utf-8', errors='ignore') as f_raw, \
         open(f'{output_dir}/corpus/{dataset_name}.clean.txt', 'w', encoding='utf-8', errors='ignore') as f_clean:
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                for item in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
                    # Get text and handle potential None values
                    text = item.get('sentence', '')
                    if text is None:
                        text = ''
                    
                    # Clean the text
                    text = str(text).replace('\n', ' ').replace('\t', ' ')
                    f_raw.write(text + '\n')
                    
                    # Simple cleaning
                    cleaned = ' '.join(text.lower().split())
                    f_clean.write(cleaned + '\n')
    
    print(f"✓ Prepared {dataset_name} dataset")
    return dataset_name

def prepare_semeval(output_dir='data'):
    """Prepare SemEval dataset"""
    print("Loading SemEval 2018 Task 3A dataset from HuggingFace...")
    dataset = load_dataset('viethq1906/semeval_2018_3A')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/corpus', exist_ok=True)
    dataset_name = 'semeval3a'
    
    # Create data files - specify utf-8 encoding
    with open(f'{output_dir}/{dataset_name}.txt', 'w', encoding='utf-8') as f:
        idx = 0
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                for item in dataset[split_name]:
                    label = 'ironic' if item['sentiment'] == 1 else 'not_ironic'
                    f.write(f"doc_{idx}\t{split_name}\t{label}\n")
                    idx += 1
    
    # Create corpus file - specify utf-8 encoding and handle potential encoding errors
    with open(f'{output_dir}/corpus/{dataset_name}.txt', 'w', encoding='utf-8', errors='ignore') as f_raw, \
         open(f'{output_dir}/corpus/{dataset_name}.clean.txt', 'w', encoding='utf-8', errors='ignore') as f_clean:
        
        for split_name in ['train', 'validation', 'test']:
            if split_name in dataset:
                for item in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
                    # Get text and handle potential None values
                    text = item.get('sentence', '')
                    if text is None:
                        text = ''
                    
                    # Clean the text
                    text = str(text).replace('\n', ' ').replace('\t', ' ')
                    f_raw.write(text + '\n')
                    
                    # Simple cleaning
                    cleaned = ' '.join(text.lower().split())
                    f_clean.write(cleaned + '\n')
    
    print(f"✓ Prepared {dataset_name} dataset")
    
    # Check for custom test set
    custom_test_path = f'{output_dir}/semeval_2018_3a_custom_test.csv'
    if os.path.exists(custom_test_path):
        print(f"✓ Custom test set found: {custom_test_path}")
    else:
        print(f"⚠ Warning: Custom test set not found at {custom_test_path}")
        print(f"  If you want to use --use_custom_test, please place the CSV file at:")
        print(f"  {custom_test_path}")
        print(f"  Expected CSV columns: sentence, sentiment")
    
    return dataset_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=['isarcasm', 'semeval3a', 'all'])
    parser.add_argument('--output_dir', type=str, default='data')
    args = parser.parse_args()
    
    if args.dataset == 'isarcasm' or args.dataset == 'all':
        prepare_isarcasm(args.output_dir)
    
    if args.dataset == 'semeval3a' or args.dataset == 'all':
        prepare_semeval(args.output_dir)
    
    print("\n✓ Dataset preparation complete!")