from prepare_twt_dataset import clean_text
import torch as th
import numpy as np
import pandas as pd
from model import BertClassifier

bert_init = 'dost-asti/RoBERTa-tl-sentiment-analysis'
nb_class = 3
batch_size = 128
label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

checkpoint_paths = [
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold0_seed42_gcn_20260327_150430/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold1_seed42_gcn_20260327_151805/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold2_seed42_gcn_20260327_152911/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold3_seed42_gcn_20260327_154129/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold4_seed42_gcn_20260327_155348/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold5_seed42_gcn_20260327_160345/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold6_seed42_gcn_20260327_161709/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold7_seed42_gcn_20260327_162929/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold8_seed42_gcn_20260327_164304/checkpoint.pth',
    'checkpoint/k=10/no-conv_m0.0/checkpoint/twitter_fold9_seed42_gcn_20260327_165853/checkpoint.pth',
]

# Load and clean unlabeled data
df = pd.read_csv('data/tweets_unlabeled_set.csv')
df['cleaned_text'] = df['text'].apply(clean_text)
texts = df['cleaned_text'].tolist()
print(f"Loaded {len(texts)} unlabeled tweets")

# Tokenize once in batches
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
model_ref = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)
input_ids, attention_mask = tokenize_in_batches(texts, model_ref.tokenizer)
del model_ref  # free memory

# Run inference for each fold checkpoint
all_probs = []

for fold_idx, ckpt_path in enumerate(checkpoint_paths):
    print(f"Running inference — fold {fold_idx + 1}/{len(checkpoint_paths)}")
    
    model = BertClassifier(pretrained_model=bert_init, nb_class=nb_class)
    ckpt = th.load(ckpt_path, map_location='cpu')
    model.bert_model.load_state_dict(ckpt['bert_model'])
    model.classifier.load_state_dict(ckpt['classifier'])
    model.eval()

    fold_probs = []
    with th.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_input_ids = input_ids[i:i+batch_size]
            batch_attention_mask = attention_mask[i:i+batch_size]
            logits = model(batch_input_ids, batch_attention_mask)
            probs = th.nn.Softmax(dim=1)(logits).cpu().numpy()
            fold_probs.append(probs)
    
    fold_probs = np.concatenate(fold_probs, axis=0)
    all_probs.append(fold_probs)
    del model  # free memory after each fold

# Ensemble: average probabilities across folds
print("Ensembling predictions...")
ensemble_probs = np.mean(all_probs, axis=0)  # shape: (n_docs, nb_class)
preds = ensemble_probs.argmax(axis=1)

# Save results
df['predicted_sentiment'] = [label_map[p] for p in preds]
df['prob_positive'] = ensemble_probs[:, 0]
df['prob_negative'] = ensemble_probs[:, 1]
df['prob_neutral']  = ensemble_probs[:, 2]

df.to_csv('predictions.csv', index=False)
print(f"Predictions saved to predictions.csv")
print(f"\nPrediction distribution:")
print(df['predicted_sentiment'].value_counts())