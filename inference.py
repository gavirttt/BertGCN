from prepare_twt_dataset import clean_text, load_author_lookup
from tqdm import tqdm
import os
import torch as th
import numpy as np
import pandas as pd
from model import BertClassifier

bert_init = 'dost-asti/RoBERTa-tl-sentiment-analysis'
nb_class = 3
batch_size = 128
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}
device = th.device('cuda' if th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint_path = 'checkpoint/dost-asti/RoBERTa-tl-sentiment-analysis_twitter/checkpoint.pth'

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
print(f"Using checkpoint: {checkpoint_path}")

# Load and clean unlabeled data
df = pd.read_csv('data/tweets_unlabeled_set.csv')
load_author_lookup("data/well_known_authors_philippine_elections.csv")
df['cleaned_text'] = df['text'].apply(clean_text)
texts = df['cleaned_text'].tolist()
print(f"Loaded {len(texts)} unlabeled tweets")

# Load model and checkpoint
print("Loading model...")
model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)
ckpt = th.load(checkpoint_path, map_location=device)
model.bert_model.load_state_dict(ckpt['bert_model'])
model.classifier.load_state_dict(ckpt['classifier'])
model.eval()
model = model.to(device)

# Tokenize in batches
def tokenize_in_batches(texts, tokenizer, max_length=128, batch_size=128):
    all_input_ids = []
    all_attention_masks = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            batch,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        all_input_ids.append(encoded['input_ids'])
        all_attention_masks.append(encoded['attention_mask'])
    return th.cat(all_input_ids, dim=0), th.cat(all_attention_masks, dim=0)

print("Tokenizing...")
input_ids, attention_mask = tokenize_in_batches(texts, model.tokenizer)
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

# Run inference
print("Running inference...")
all_probs = []
with th.no_grad():
    for i in tqdm(range(0, len(texts), batch_size), desc='Inference', unit='batch'):
        batch_input_ids = input_ids[i:i+batch_size]
        batch_attention_mask = attention_mask[i:i+batch_size]
        logits = model(batch_input_ids, batch_attention_mask)
        probs = th.nn.Softmax(dim=1)(logits).cpu().numpy()
        all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)
preds = all_probs.argmax(axis=1)

# Save results
df['sentiment'] = [label_map[p] for p in preds]
df['prob_positive'] = all_probs[:, 0]
df['prob_negative'] = all_probs[:, 1]
df['prob_neutral']  = all_probs[:, 2]

df.to_csv('predictions.csv', index=False)
print(f"Predictions saved to predictions.csv")
print(f"\nPrediction distribution:")
print(df['sentiment'].value_counts())