import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import argparse, os, json
from datetime import datetime

# ----------------- usage -----------------
# Basic usage (uses top 10 words, window size 10):
#   python coherence_scoring.py \
#       --clusters tweets_with_clusters.csv \
#       --keywords llr_keywords.json
#
# Custom top-N and window size:
#   python coherence_scoring.py \
#       --clusters tweets_with_clusters.csv \
#       --keywords llr_keywords.json \
#       --top_n 10 \
#       --window_size 10
#
# Run sensitivity analysis (sweeps top_n from 5 to 20):
#   python coherence_scoring.py --clusters results/all-sentiment/all-months/tweets_with_clusters.csv --keywords results/all-sentiment/all-months/llr_keywords.json --sensitivity
# ----------------- ----- -----------------

parser = argparse.ArgumentParser()
parser.add_argument('--clusters', type=str, required=True,
                    help='Path to tweets_with_clusters.csv (output of topic_modeling.py)')
parser.add_argument('--keywords', type=str, required=True,
                    help='Path to llr_keywords.json (output of topic_modeling.py)')
parser.add_argument('--top_n', type=int, default=10,
                    help='Number of top words per cluster to use for coherence scoring (default: 10)')
parser.add_argument('--window_size', type=int, default=10,
                    help='Sliding window size for UCI co-occurrence (default: 10)')
parser.add_argument('--sensitivity', action='store_true',
                    help='Run sensitivity analysis sweeping top_n from 5 to 20')
parser.add_argument('--output_dir', type=str,
                    default=f'coherence_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                    help='Directory to save outputs')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Load inputs ───────────────────────────────────────────────────────────────
print(f"Loading cluster CSV: {args.clusters}")
df = pd.read_csv(args.clusters)
print(f"  {len(df)} tweets, columns: {df.columns.tolist()}")

print(f"Loading LLR keywords: {args.keywords}")
with open(args.keywords, 'r', encoding='utf-8') as f:
    llr_results = json.load(f)

n_topics = len(llr_results)
print(f"  {n_topics} clusters found: {list(llr_results.keys())}")

# ── Resolve text column ───────────────────────────────────────────────────────
if 'cleaned_text2' in df.columns:
    text_col = 'cleaned_text2'
elif 'cleaned_text' in df.columns:
    text_col = 'cleaned_text'
else:
    text_col = 'text'
print(f"  Using text column: '{text_col}'")
texts_cleaned = df[text_col].fillna('').tolist()

# ── Build CountVectorizer (same settings as topic_modeling.py) ────────────────
tagalog_stopwords = [
    'ng', 'sa', 'ang', 'na', 'ni', 'si',
    'mga', 'at', 'ay', 'ko', 'mo', 'ka',
    'po', 'naman', 'lang', 'din', 'rin',
    'yung', 'ung', 'sya', 'siya', 'niya',
    'nila', 'namin', 'kami', 'kayo', 'sila',
    'ako', 'ikaw', 'ito', 'iyon', 'dito',
    'doon', 'para', 'pero', 'kasi', 'kaya',
    'kung', 'pag', 'kapag', 'dahil', 'nang',
    'hindi', 'wala', 'may', 'yun', 'yan',
    'yon', 'ba', 'nga', 'eh', 'oh', 'talaga',
    'ganun', 'ganon', 'parang', 'daw', 'raw',
    'opo', 'oo', 'umano', 'natin', 'ayon',
    'nya', 'ating', 'mas', 'atin', 'niyo',
    'ninyo', 'akin', 'amin', 'ano', 'nito',
    'gayunman', 'inyo', 'iyo', 'kanya',
    'kaniya', 'kanila', 'kanino', 'mismo',
    'narito', 'nandito', 'rito', 'ng', 'sang',
    'ang', 'nang', 'mga', 'at', 'ay', 'kong',
    'mong', 'kang', 'pong', 'namang', 'lang',
    'ding', 'ring', 'yung', 'ung', 'syang',
    'siyang', 'niyang', 'nilang', 'naming',
    'kaming', 'kayong', 'silang', 'akong',
    'ikaw', 'itong', 'iyong', 'ditong', 'doong',
    'parang', 'pero', 'kasing', 'kayang',
    'kung', 'pag', 'kapag', 'dahil', 'nang',
    'hinding', 'walang', 'may', 'yung', 'yang',
    'yong', 'bang', 'ngang', 'eh', 'oh',
    'talagang', 'ganung', 'ganong', 'parang',
    'daw', 'raw', 'opo', 'oo', 'umanong',
    'nating', 'ayong', 'nyang', 'ating', 'mas',
    'ating', 'niyong', 'ninyong', 'aking', 'aming',
    'anong', 'nitong', 'gayunmang', 'inyong',
    'iyong', 'kanyang', 'kaniyang', 'kanilang',
    'kaninong', 'mismong', 'naritong', 'nanditong',
    'ritong', 'nyong', 'saan', 'saang'
]
english_stopwords = list(CountVectorizer(stop_words='english').get_stop_words())
all_stopwords = list(set(english_stopwords + tagalog_stopwords))

print("\nFitting CountVectorizer...")
vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words=all_stopwords
)
doc_term_matrix = vectorizer.fit_transform(texts_cleaned)
vocab       = vectorizer.get_feature_names_out().tolist()
word2id     = {w: i for i, w in enumerate(vocab)}
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Doc-term matrix: {doc_term_matrix.shape}")

# ── Helper: get top-N words from LLR results ──────────────────────────────────
def get_top_words(llr_results, cluster_key, top_n):
    keywords = llr_results[cluster_key]
    return [w for w, _ in keywords[:top_n]]

# ── UMass ─────────────────────────────────────────────────────────────────────
def build_cooccurrence(doc_term_matrix):
    """Binary presence matrix and per-word document frequency."""
    binary    = (doc_term_matrix > 0).astype(np.float32)
    doc_freq  = np.asarray(binary.sum(axis=0)).flatten()
    return binary, doc_freq

def compute_umass(top_words, binary_matrix, doc_freq, word2id, epsilon=1.0):
    """
    UMass coherence:
      C_UMass = (2 / N(N-1)) * sum_{j>i} log( (D(wi,wj) + eps) / D(wj) )
    Less negative = more coherent.
    """
    N = len(top_words)
    if N < 2:
        return float('nan')
    score, pairs = 0.0, 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            wi, wj = top_words[i], top_words[j]
            if wi not in word2id or wj not in word2id:
                continue
            idx_i = word2id[wi]
            idx_j = word2id[wj]
            col_i   = np.asarray(binary_matrix[:, idx_i].todense()).flatten()
            col_j   = np.asarray(binary_matrix[:, idx_j].todense()).flatten()
            d_wi_wj = float(np.logical_and(col_i, col_j).sum())
            d_wj    = float(doc_freq[idx_j])
            if d_wj == 0:
                continue
            score += np.log((d_wi_wj + epsilon) / d_wj)
            pairs += 1
    if pairs == 0:
        return float('nan')
    return (2.0 / (N * (N - 1))) * score

# ── UCI ───────────────────────────────────────────────────────────────────────
def build_uci_cooccurrence(texts, word2id, window_size):
    """Sliding-window co-occurrence counts for UCI."""
    word_count, pair_count, total_windows = {}, {}, 0
    for text in texts:
        tokens    = text.lower().split()
        token_ids = [word2id[t] for t in tokens if t in word2id]
        n         = len(token_ids)
        for start in range(n):
            window = set(token_ids[start: start + window_size])
            total_windows += 1
            for wid in window:
                word_count[wid] = word_count.get(wid, 0) + 1
            win_list = list(window)
            for a in range(len(win_list)):
                for b in range(a + 1, len(win_list)):
                    pair = (min(win_list[a], win_list[b]),
                            max(win_list[a], win_list[b]))
                    pair_count[pair] = pair_count.get(pair, 0) + 1
    return word_count, pair_count, total_windows

def compute_uci(top_words, word_count, pair_count, total_windows, word2id, epsilon=1.0):
    """
    UCI coherence:
      C_UCI = (2 / N(N-1)) * sum_{j>i} log( (P(wi,wj) + eps) / (P(wi) * P(wj)) )
    Less negative = more coherent.
    """
    N = len(top_words)
    if N < 2:
        return float('nan')
    score, pairs = 0.0, 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            wi, wj = top_words[i], top_words[j]
            if wi not in word2id or wj not in word2id:
                continue
            idx_i  = word2id[wi]
            idx_j  = word2id[wj]
            cnt_i  = word_count.get(idx_i, 0)
            cnt_j  = word_count.get(idx_j, 0)
            pair   = (min(idx_i, idx_j), max(idx_i, idx_j))
            cnt_ij = pair_count.get(pair, 0)
            if cnt_i == 0 or cnt_j == 0 or total_windows == 0:
                continue
            p_wi    = cnt_i  / total_windows
            p_wj    = cnt_j  / total_windows
            p_wi_wj = cnt_ij / total_windows
            score  += np.log((p_wi_wj + epsilon / total_windows) / (p_wi * p_wj))
            pairs  += 1
    if pairs == 0:
        return float('nan')
    return (2.0 / (N * (N - 1))) * score

# ── Build shared structures ───────────────────────────────────────────────────
print("\nBuilding co-document matrix for UMass...")
binary_matrix, doc_freq = build_cooccurrence(doc_term_matrix)

print(f"Building sliding-window co-occurrence for UCI (window={args.window_size})...")
word_count, pair_count, total_windows = build_uci_cooccurrence(
    texts_cleaned, word2id, window_size=args.window_size
)

# ── Score each cluster ────────────────────────────────────────────────────────
def score_all_clusters(llr_results, top_n):
    results = {}
    for cluster_key in llr_results:
        top_words = get_top_words(llr_results, cluster_key, top_n)
        umass = compute_umass(top_words, binary_matrix, doc_freq, word2id)
        uci   = compute_uci(top_words, word_count, pair_count, total_windows, word2id)
        results[cluster_key] = {
            'top_words': top_words,
            'umass':     round(umass, 6) if not np.isnan(umass) else None,
            'uci':       round(uci,   6) if not np.isnan(uci)   else None,
        }
    return results

# ── Main coherence run ────────────────────────────────────────────────────────
print(f"\nScoring coherence with top_n={args.top_n}...")
coherence_results = score_all_clusters(llr_results, args.top_n)

for cluster_key, vals in coherence_results.items():
    print(f"  {cluster_key}: UMass={vals['umass']:.4f}  UCI={vals['uci']:.4f}"
          f"  | words: {vals['top_words'][:5]}")

avg_umass = np.nanmean([v['umass'] for v in coherence_results.values()])
avg_uci   = np.nanmean([v['uci']   for v in coherence_results.values()])
print(f"\nAverage UMass : {avg_umass:.4f}")
print(f"Average UCI   : {avg_uci:.4f}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
json_path = os.path.join(args.output_dir, 'coherence_scores.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(coherence_results, f, indent=2, ensure_ascii=False)
print(f"\nJSON saved to {json_path}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
rows = [
    {
        'cluster':   k,
        'umass':     v['umass'],
        'uci':       v['uci'],
        'top_words': ', '.join(v['top_words']),
    }
    for k, v in coherence_results.items()
]
rows.append({'cluster': 'AVERAGE', 'umass': round(avg_umass, 6),
             'uci': round(avg_uci, 6), 'top_words': ''})
csv_path = os.path.join(args.output_dir, 'coherence_scores.csv')
pd.DataFrame(rows).to_csv(csv_path, index=False)
print(f"CSV saved to {csv_path}")

# ── Bar chart ─────────────────────────────────────────────────────────────────
def plot_coherence_bars(coherence_results, avg_umass, avg_uci, output_dir, top_n):
    cluster_keys = list(coherence_results.keys())
    umass_scores = [coherence_results[c]['umass'] or 0 for c in cluster_keys]
    uci_scores   = [coherence_results[c]['uci']   or 0 for c in cluster_keys]
    x, width     = np.arange(len(cluster_keys)), 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x, umass_scores, width, color='#377eb8', alpha=0.85)
    ax1.axhline(avg_umass, color='#e41a1c', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_umass:.3f}')
    ax1.set_title('UMass Coherence per Cluster', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('UMass Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(cluster_keys, rotation=15)
    ax1.legend()
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 h + abs(h) * 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=9)

    bars2 = ax2.bar(x, uci_scores, width, color='#4daf4a', alpha=0.85)
    ax2.axhline(avg_uci, color='#e41a1c', linestyle='--', linewidth=1.5,
                label=f'Avg: {avg_uci:.3f}')
    ax2.set_title('UCI Coherence per Cluster', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('UCI Score')
    ax2.set_xticks(x)
    ax2.set_xticklabels(cluster_keys, rotation=15)
    ax2.legend()
    for bar in bars2:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 h + abs(h) * 0.01, f'{h:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Topic Coherence Scores (top_n={top_n})', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, 'coherence_scores.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar chart saved to {path}")

plot_coherence_bars(coherence_results, avg_umass, avg_uci, args.output_dir, args.top_n)

# ── Sensitivity analysis ──────────────────────────────────────────────────────
if args.sensitivity:
    print("\nRunning sensitivity analysis (top_n = 5 to 20)...")
    n_range    = range(5, 21)
    cluster_keys = list(llr_results.keys())
    sensitivity  = {k: {'umass': [], 'uci': []} for k in cluster_keys}

    for n in n_range:
        res = score_all_clusters(llr_results, top_n=n)
        for k in cluster_keys:
            sensitivity[k]['umass'].append(res[k]['umass'] or 0)
            sensitivity[k]['uci'].append(res[k]['uci']   or 0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for k in cluster_keys:
        axes[0].plot(list(n_range), sensitivity[k]['umass'], marker='o', label=k)
        axes[1].plot(list(n_range), sensitivity[k]['uci'],   marker='o', label=k)

    for ax, metric in zip(axes, ['UMass', 'UCI']):
        ax.axvline(args.top_n, color='gray', linestyle='--', linewidth=1,
                   label=f'current top_n={args.top_n}')
        ax.set_title(f'{metric} Coherence vs top_n', fontsize=12, fontweight='bold')
        ax.set_xlabel('top_n')
        ax.set_ylabel(f'{metric} Score')
        ax.set_xticks(list(n_range))
        ax.legend(fontsize=8)

    plt.suptitle('Sensitivity Analysis: Coherence vs Number of Top Words',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    sens_path = os.path.join(args.output_dir, 'coherence_sensitivity.png')
    plt.savefig(sens_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sensitivity plot saved to {sens_path}")

    # Save sensitivity CSV
    sens_rows = []
    for n in n_range:
        row = {'top_n': n}
        res = score_all_clusters(llr_results, top_n=n)
        for k in cluster_keys:
            row[f'{k}_umass'] = res[k]['umass']
            row[f'{k}_uci']   = res[k]['uci']
        row['avg_umass'] = round(np.nanmean([res[k]['umass'] for k in cluster_keys]), 6)
        row['avg_uci']   = round(np.nanmean([res[k]['uci']   for k in cluster_keys]), 6)
        sens_rows.append(row)
    sens_csv = os.path.join(args.output_dir, 'coherence_sensitivity.csv')
    pd.DataFrame(sens_rows).to_csv(sens_csv, index=False)
    print(f"Sensitivity CSV saved to {sens_csv}")

print(f"\n✓ Done. All outputs saved to {args.output_dir}/")