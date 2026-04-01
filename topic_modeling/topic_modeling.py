import torch as th
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from wordcloud import WordCloud
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse, os, json
from datetime import datetime
 
# ----------------- usage -----------------
# # All tweets, all months
# python topic_modeling.py --csv data/tweets_predictions.csv --n_topics 5 --device cuda
 
# # Per sentiment
# python topic_modeling.py --csv data/tweets_predictions.csv --sentiment negative --n_topics 5 --device cuda
 
# # Per sentiment, per month
# python topic_modeling.py --csv data/tweets_predictions.csv --sentiment negative --month 2024-01 --n_topics 5 --device cuda
 
# # Labeled CSV (integer sentiments 0/1/2)
# python topic_modeling.py --csv data/tweets_labeled_set.csv --sentiment 0 --n_topics 5 --device cuda
# ----------------- ----- -----------------
 
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, required=True,
                    help='Labeled CSV (sentiment as 0/1/2) or '
                         'predictions CSV (sentiment as positive/negative/neutral)')
parser.add_argument('--bert_init', type=str,
                    default='dost-asti/RoBERTa-tl-sentiment-analysis',
                    help='Pretrained BERT model to use for embeddings')
parser.add_argument('--n_topics', type=int, default=5)
parser.add_argument('--sentiment', type=str, default=None,
                    help='Filter by sentiment. ',
                    choices=['positive', 'negative', 'neutral'])
parser.add_argument('--month', type=str, default=None,
                    help='Filter by month. Omit for all months.',
                    choices=['2024-12', '2025-01', '2025-02', '2025-03', '2025-04', '2025-05'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_length', type=int, default=128)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--output_dir', type=str, default=f'topic_modeling/results_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
args = parser.parse_args()
 
os.makedirs(args.output_dir, exist_ok=True)
device = th.device('cuda' if args.device == 'cuda' and th.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
 
colors = [
    '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    '#a65628', '#f781bf', '#999999', '#dede00', '#17becf'
]
 
label_map_int = {0: 'positive', 1: 'negative', 2: 'neutral'}
 
# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
print(f"Loaded {len(df)} tweets from {args.csv}")
print(f"Columns: {df.columns.tolist()}")
 
# ── Parse dates ───────────────────────────────────────────────────────────────
if 'createdAt' in df.columns:
    df['createdAt'] = pd.to_datetime(df['createdAt'])
    df['year_month'] = df['createdAt'].dt.to_period('M')
    print(f"\nDate range: {df['year_month'].min()} to {df['year_month'].max()}")
    print(f"Months available:\n{df['year_month'].value_counts().sort_index().to_string()}")
else:
    print("WARNING: 'createdAt' column not found — monthly analysis unavailable.")
    df['year_month'] = None
 
# ── Normalize sentiment column ────────────────────────────────────────────────
sentiment_col = 'sentiment'
if df[sentiment_col].dtype in ['int64', 'float64']:
    # labeled CSV — map integers to string labels
    df['sentiment_label'] = df[sentiment_col].map(label_map_int)
else:
    # predictions CSV — already strings
    df['sentiment_label'] = df[sentiment_col].str.lower().str.strip()
 
print(f"\nSentiment distribution:\n{df['sentiment_label'].value_counts().to_string()}")
 
# ── Monthly sentiment trend (full dataset, before any filtering) ──────────────
def plot_monthly_sentiment(df, output_dir):
    """Plot sentiment counts and proportions over time."""
    if df['year_month'].isnull().all():
        print("Skipping monthly trend — no date column.")
        return
 
    monthly = (
        df.groupby(['year_month', 'sentiment_label'])
        .size()
        .unstack(fill_value=0)
    )
    monthly_pct = monthly.div(monthly.sum(axis=1), axis=0) * 100
 
    sentiment_colors = {
        'positive': '#4daf4a',
        'negative': '#e41a1c',
        'neutral':  '#377eb8'
    }
    plot_colors = [sentiment_colors.get(c, '#999999') for c in monthly.columns]
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9))
 
    monthly.plot(kind='bar', ax=ax1, color=plot_colors, width=0.8)
    ax1.set_title('Monthly Sentiment Counts', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Tweet Count')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Sentiment')
 
    monthly_pct.plot(kind='bar', ax=ax2, color=plot_colors, width=0.8)
    ax2.set_title('Monthly Sentiment Distribution (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Percentage')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(title='Sentiment')
 
    plt.tight_layout()
    trend_dir = os.path.join(output_dir, 'all-sentiment', 'all-months')
    os.makedirs(trend_dir, exist_ok=True)
    path = os.path.join(trend_dir, 'monthly_sentiment_trend.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nMonthly sentiment trend saved to {path}")
 
    monthly_csv = os.path.join(trend_dir, 'monthly_sentiment_counts.csv')
    monthly.to_csv(monthly_csv)
    print(f"Monthly counts saved to {monthly_csv}")
 
if not args.sentiment and not args.month:
    plot_monthly_sentiment(df, args.output_dir)
 
# ── Apply sentiment filter ────────────────────────────────────────────────────
if args.sentiment:
    sentiment_filter = args.sentiment.lower().strip()
    available_sentiments = df['sentiment_label'].unique().tolist()  # capture before filtering
    df = df[df['sentiment_label'] == sentiment_filter].reset_index(drop=True)
    print(f"\nFiltered to sentiment='{sentiment_filter}': {len(df)} tweets")
    if len(df) == 0:
        raise ValueError(
            f"No tweets found for sentiment='{args.sentiment}'. "
            f"Available: {available_sentiments}"
        )
 
# ── Apply month filter ────────────────────────────────────────────────────────
if args.month:
    df = df[df['year_month'].astype(str) == args.month].reset_index(drop=True)
    print(f"Filtered to month='{args.month}': {len(df)} tweets")
    if len(df) == 0:
        raise ValueError(
            f"No tweets found for month='{args.month}'. "
            f"Check --month format (e.g. 2024-01)."
        )
 
# ── Build output tag for filenames ────────────────────────────────────────────
sentiment_tag = f'{args.sentiment}-sentiment' if args.sentiment else 'all-sentiment'
month_tag     = args.month or 'all-months'
output_subdir    = f'{sentiment_tag}/{month_tag}/'
os.makedirs(os.path.join(args.output_dir, output_subdir), exist_ok=True)
print(f"\nOutput sub-directory: {args.output_dir}/{output_subdir}")
print(f"Total tweets for analysis: {len(df)}")
 
# Use cleaned_text for both BERT and LDA
texts_cleaned = df['cleaned_text2'].fillna('').tolist() \
    if 'cleaned_text2' in df.columns \
    else df['text'].fillna('').tolist()  # fallback if cleaned_text missing
 
# ── Load BERT (pretrained, no checkpoint needed) ──────────────────────────────
print(f"\nLoading BERT: {args.bert_init}")
tokenizer  = AutoTokenizer.from_pretrained(args.bert_init)
bert_model = AutoModel.from_pretrained(args.bert_init)
bert_model = bert_model.to(device)
bert_model.eval()
print("Using pretrained model as-is (no fine-tuned checkpoint)")
 
# ── Extract BERT CLS embeddings ───────────────────────────────────────────────
print("\nExtracting BERT embeddings...")
all_embeddings = []
 
for i in tqdm(range(0, len(texts_cleaned), args.batch_size),
              desc='BERT embeddings', unit='batch'):
    batch = texts_cleaned[i:i + args.batch_size]
    encoded = tokenizer(
        batch,
        max_length=args.max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    with th.no_grad():
        output = bert_model(
            input_ids=encoded['input_ids'].to(device),
            attention_mask=encoded['attention_mask'].to(device)
        )[0][:, 0]  # CLS token
    all_embeddings.append(output.cpu().numpy())
 
bert_embeddings = np.vstack(all_embeddings)
print(f"BERT embeddings shape: {bert_embeddings.shape}")
 
# ── LDA topic distributions ───────────────────────────────────────────────────
print("\nFitting LDA...")
 
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
    'ritong', 'nyong', 'saan', 'saang', 'upang',
    'tapos', 'mula'
]
 
english_stopwords = list(CountVectorizer(stop_words='english').get_stop_words())
all_stopwords = list(set(english_stopwords + tagalog_stopwords))
 
vectorizer = CountVectorizer(
    max_features=5000,
    min_df=2,
    max_df=0.95,
    stop_words=all_stopwords
)
doc_term_matrix = vectorizer.fit_transform(texts_cleaned)
 
lda = LatentDirichletAllocation(
    n_components=args.n_topics,
    random_state=42,
    max_iter=50,
    learning_method='batch'
)
lda_embeddings = lda.fit_transform(doc_term_matrix)
print(f"LDA embeddings shape: {lda_embeddings.shape}")
 
# ── Joint BERT-LDA embeddings ─────────────────────────────────────────────────
bert_norm        = normalize(bert_embeddings)
lda_norm         = normalize(lda_embeddings)
joint_embeddings = np.hstack([bert_norm, lda_norm])
print(f"Joint embeddings shape: {joint_embeddings.shape}")
 
# ── K-means clustering ────────────────────────────────────────────────────────
print(f"\nRunning K-means with k={args.n_topics}...")
kmeans         = KMeans(n_clusters=args.n_topics, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(joint_embeddings)
df['cluster']  = cluster_labels
 
# ── UMAP visualization ────────────────────────────────────────────────────────
print("Running UMAP...")
reducer         = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
umap_embeddings = reducer.fit_transform(joint_embeddings)
 
plt.figure(figsize=(10, 8))
for i in range(args.n_topics):
    mask = cluster_labels == i
    pct  = mask.sum() / len(cluster_labels) * 100
    plt.scatter(
        umap_embeddings[mask, 0],
        umap_embeddings[mask, 1],
        c=colors[i % len(colors)],
        label=f'Cluster {i}: {pct:.1f}%',
        alpha=0.5,
        s=5
    )
plt.legend(markerscale=3)
plt.title(f'UMAP — Topic Clusters ({output_subdir})', fontsize=13)
umap_path = os.path.join(args.output_dir, f'{output_subdir}umap.png')
plt.savefig(umap_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"UMAP saved to {umap_path}")
 
# ── LLR keyword extraction ────────────────────────────────────────────────────
print("\nComputing LLR keywords per cluster...")
feature_names = vectorizer.get_feature_names_out()
 
def compute_llr(doc_term_matrix, cluster_labels, cluster_id, top_n=30):
    """Log-Likelihood Ratio for words in a cluster vs the rest."""
    in_cluster  = (cluster_labels == cluster_id)
    out_cluster = ~in_cluster

    # Use sparse boolean operations to avoid dense OOM on large matrices
    in_docs_sparse  = doc_term_matrix[in_cluster]
    out_docs_sparse = doc_term_matrix[out_cluster]

    N_in  = in_docs_sparse.shape[0]
    N_out = out_docs_sparse.shape[0]
    N     = N_in + N_out
    eps   = 1e-10

    # Binary word presence counts computed sparsely
    k_arr = np.asarray((in_docs_sparse  > 0).sum(axis=0)).flatten()
    l_arr = np.asarray((out_docs_sparse > 0).sum(axis=0)).flatten()
    m_arr = N_in  - k_arr
    n_arr = N_out - l_arr

    llr_scores = []
    for j in range(doc_term_matrix.shape[1]):
        k = k_arr[j]; l = l_arr[j]; m = m_arr[j]; n = n_arr[j]

        p_w    = (k + l) / N
        p_w_T  = k / N_in  if N_in  > 0 else eps
        p_w_nT = l / N_out if N_out > 0 else eps
 
        try:
            llr = 2 * (
                k * np.log(p_w_T   / (p_w + eps) + eps) +
                l * np.log(p_w_nT  / (p_w + eps) + eps) +
                m * np.log((1 - p_w_T)  / (1 - p_w + eps) + eps) +
                n * np.log((1 - p_w_nT) / (1 - p_w + eps) + eps)
            )
        except Exception:
            llr = 0.0
 
        llr_scores.append(llr)
 
    top_indices = np.argsort(llr_scores)[::-1][:top_n]
    return [(feature_names[i], llr_scores[i]) for i in top_indices]
 
 
llr_results = {}
for c in range(args.n_topics):
    keywords = compute_llr(doc_term_matrix, cluster_labels, c)
    llr_results[f'cluster_{c}'] = keywords
    print(f"\nCluster {c} top keywords:")
    print([w for w, _ in keywords[:15]])
 
# ── Word clouds ───────────────────────────────────────────────────────────────
# ── Sentiment-colored word clouds using existing tweet labels ──────────────────
def compute_word_sentiment_distribution(words, texts_cleaned, df, cluster_labels, cluster_id):
    """
    Compute sentiment distribution for each word based on the tweets it appears in.
    """
    # Get indices of tweets in this cluster
    cluster_indices = [i for i in range(len(cluster_labels)) if cluster_labels[i] == cluster_id]
    
    # Get sentiment labels for these tweets
    cluster_df = df.iloc[cluster_indices].copy()
    cluster_texts = [texts_cleaned[i] for i in cluster_indices]
    
    word_sentiment_counts = {}
    
    for word in words:
        # Initialize counts for this word
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        }
        
        # Find all tweets containing this word
        for idx, text in enumerate(cluster_texts):
            if word.lower() in text.lower():
                sentiment = cluster_df.iloc[idx]['sentiment_label']
                sentiment_counts[sentiment] += 1
                sentiment_counts['total'] += 1
        
        # Calculate proportions
        if sentiment_counts['total'] > 0:
            proportions = {
                'positive': sentiment_counts['positive'] / sentiment_counts['total'],
                'negative': sentiment_counts['negative'] / sentiment_counts['total'],
                'neutral': sentiment_counts['neutral'] / sentiment_counts['total']
            }
            # Determine dominant sentiment
            dominant = max(proportions, key=proportions.get)
            confidence = proportions[dominant]
        else:
            proportions = {'positive': 0, 'negative': 0, 'neutral': 0}
            dominant = 'neutral'
            confidence = 0.5
        
        word_sentiment_counts[word] = {
            'counts': sentiment_counts,
            'proportions': proportions,
            'dominant': dominant,
            'confidence': confidence
        }
    
    return word_sentiment_counts

def sentiment_color_func_tweet_based(word, word_sentiment_data, font_size=None, 
                                     position=None, orientation=None, font_path=None, 
                                     random_state=None):
    """
    Color function for word cloud based on sentiment distribution from tweet labels.
    """
    sentiment_colors = {
        'positive': '#4daf4a',
        'negative': '#e41a1c',
        'neutral':  "#777777"
    }

    if word not in word_sentiment_data:
        return '#999999'

    dominant_sentiment = word_sentiment_data[word]['dominant']
    confidence = word_sentiment_data[word]['confidence']

    # Parse hex → RGB
    hex_color = sentiment_colors[dominant_sentiment].lstrip('#')
    r, g, b = (int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    # Blend toward gray based on confidence (0.5 → very gray, 1.0 → full color)
    blend = max((confidence - 0.5) / 0.5, 0.0)  # normalizes to 0.0–1.0
    gray = 180
    r = int(gray + (r - gray) * blend)
    g = int(gray + (g - gray) * blend)
    b = int(gray + (b - gray) * blend)

    return f'#{r:02x}{g:02x}{b:02x}'

def generate_sentiment_wordclouds_tweet_based(llr_results, texts_cleaned, df, 
                                              cluster_labels, output_dir, 
                                              output_subdir, n_topics):
    """
    Generate word clouds with sentiment coloring based on tweet-level sentiment labels.
    """
    print("\nGenerating sentiment-colored word clouds using tweet-level sentiment labels...")
    
    # Create directory for sentiment word clouds
    sentiment_wc_dir = os.path.join(output_dir, f'{output_subdir}sentiment_wordclouds')
    os.makedirs(sentiment_wc_dir, exist_ok=True)
    
    # Store overall sentiment statistics
    all_cluster_stats = {}
    
    # Process each cluster
    for c in range(n_topics):
        print(f"\nProcessing Cluster {c}...")
        
        # Get keywords for this cluster
        keywords = llr_results[f'cluster_{c}']
        word_weights = {w: s for w, s in keywords if s > 0}
        
        if not word_weights:
            print(f"  Cluster {c}: no positive LLR scores, skipping.")
            continue
        
        # Get unique words from this cluster
        cluster_words = list(word_weights.keys())
        
        # Compute sentiment distribution for each word based on tweet labels
        word_sentiment_data = compute_word_sentiment_distribution(
            cluster_words, texts_cleaned, df, cluster_labels, c
        )
        
        # Create custom color function (default arg captures current value, not reference)
        def color_func(word, *args, wsd=word_sentiment_data, **kwargs):
            return sentiment_color_func_tweet_based(word, wsd)
        
        # Generate individual word cloud for this cluster
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            color_func=color_func,
            max_words=100,
            prefer_horizontal=0.9,
            random_state=42,
            collocations=False
        )
        wc.generate_from_frequencies(word_weights)
        
        # Save individual word cloud
        path = os.path.join(sentiment_wc_dir, f'cluster{c}_sentiment_wordcloud.png')
        wc.to_file(path)
        print(f"  Sentiment-colored word cloud saved to {path}")
        
        # Compute and save sentiment statistics for this cluster
        cluster_stats = {
            'cluster_id': c,
            'total_keywords': len(cluster_words),
            'word_sentiment_distribution': {},
            'overall_word_sentiment': {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        }
        
        for word, data in word_sentiment_data.items():
            cluster_stats['word_sentiment_distribution'][word] = {
                'dominant_sentiment': data['dominant'],
                'confidence': data['confidence'],
                'counts': data['counts'],
                'proportions': data['proportions']
            }
            cluster_stats['overall_word_sentiment'][data['dominant']] += 1
        
        # Print summary
        print(f"  Word sentiment distribution: {cluster_stats['overall_word_sentiment']}")
        
        # Create a bar chart for word sentiment distribution in this cluster
        fig, ax = plt.subplots(figsize=(8, 6))
        sentiments = ['positive', 'negative', 'neutral']
        counts = [cluster_stats['overall_word_sentiment'][s] for s in sentiments]
        colors = ['#4daf4a', '#e41a1c', '#377eb8']
        
        bars = ax.bar(sentiments, counts, color=colors)
        ax.set_title(f'Cluster {c}: Word Sentiment Distribution\n(Based on Tweet Labels)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Keywords')
        ax.set_xlabel('Sentiment')
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        stats_path = os.path.join(sentiment_wc_dir, f'cluster{c}_sentiment_distribution.png')
        plt.savefig(stats_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        all_cluster_stats[c] = cluster_stats
    
    # Create combined grid of sentiment-colored word clouds
    cols = 2
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Re-process for combined visualization (reuse cached data from first pass)
    for c in range(n_topics):
        keywords = llr_results[f'cluster_{c}']
        word_weights = {w: s for w, s in keywords if s > 0}
        
        if word_weights:
            # Reuse cached word_sentiment_data stored in all_cluster_stats
            cached_wsd = {
                word: {
                    'dominant': stats['dominant_sentiment'],
                    'confidence': stats['confidence']
                }
                for word, stats in all_cluster_stats.get(c, {})
                    .get('word_sentiment_distribution', {}).items()
            } if c in all_cluster_stats else {}

            def color_func(word, *args, wsd=cached_wsd, **kwargs):
                return sentiment_color_func_tweet_based(word, wsd)
            
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                color_func=color_func,
                max_words=100,
                prefer_horizontal=0.9,
                random_state=42,
                collocations=False
            )
            wc.generate_from_frequencies(word_weights)
            
            axes[c].imshow(wc, interpolation='bilinear')
            
            # Add sentiment distribution info as text
            if c in all_cluster_stats:
                stats = all_cluster_stats[c]['overall_word_sentiment']
                info_text = f"Positive: {stats['positive']} | Negative: {stats['negative']} | Neutral: {stats['neutral']}"
                axes[c].text(0.5, -0.05, info_text, transform=axes[c].transAxes,
                           ha='center', fontsize=8, fontweight='bold')
        
        axes[c].set_title(f'Cluster {c}', fontsize=13, fontweight='bold')
        axes[c].axis('off')
        
        # Add sentiment legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4daf4a', label='Positive words'),
            Patch(facecolor='#e41a1c', label='Negative words'),
            Patch(facecolor='#377eb8', label='Neutral words'),
            Patch(facecolor='#888888', label='Mixed/Uncertain')
        ]
        axes[c].legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    # Hide empty subplots
    for c in range(n_topics, len(axes)):
        axes[c].axis('off')
    
    plt.suptitle(f'Topic Word Clouds with Sentiment Coloring\n(Based on Tweet-Level Sentiment Labels)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    combined_path = os.path.join(sentiment_wc_dir, 'all_clusters_sentiment_wordclouds.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCombined sentiment-colored word cloud grid saved to {combined_path}")
    
    # Save all statistics as JSON
    stats_json_path = os.path.join(sentiment_wc_dir, 'word_sentiment_statistics.json')
    # Convert to serializable format
    serializable_stats = {}
    for cluster_id, stats in all_cluster_stats.items():
        serializable_stats[f'cluster_{cluster_id}'] = {
            'cluster_id': stats['cluster_id'],
            'total_keywords': stats['total_keywords'],
            'overall_word_sentiment': stats['overall_word_sentiment'],
            'word_sentiment_distribution': {}
        }
        for word, word_stats in stats['word_sentiment_distribution'].items():
            serializable_stats[f'cluster_{cluster_id}']['word_sentiment_distribution'][word] = {
                'dominant_sentiment': word_stats['dominant_sentiment'],
                'confidence': word_stats['confidence'],
                'counts': word_stats['counts'],
                'proportions': word_stats['proportions']
            }
    
    with open(stats_json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
    print(f"Word sentiment statistics saved to {stats_json_path}")
    
    return all_cluster_stats

# Generate sentiment-colored word clouds using tweet-level sentiment labels
word_sentiment_stats = generate_sentiment_wordclouds_tweet_based(
    llr_results, 
    texts_cleaned, 
    df,  # This should have 'sentiment_label' column
    cluster_labels, 
    args.output_dir, 
    output_subdir, 
    args.n_topics
)
 
# ── Save outputs ──────────────────────────────────────────────────────────────
clusters_csv = os.path.join(args.output_dir, f'{output_subdir}tweets_with_clusters.csv')
df.to_csv(clusters_csv, index=False)
print(f"\nCluster assignments saved to {clusters_csv}")
 
keywords_json = os.path.join(args.output_dir, f'{output_subdir}llr_keywords.json')
with open(keywords_json, 'w', encoding='utf-8') as f:
    json.dump(
        {k: [(w, float(s)) for w, s in v] for k, v in llr_results.items()},
        f,
        indent=2,
        ensure_ascii=False
    )
print(f"LLR keywords saved to {keywords_json}")
 
# ── Cluster x Sentiment analysis ──────────────────────────────────────────────
def analyze_cluster_sentiment(df, cluster_labels, output_dir, output_subdir, n_topics, colors):
    """Compute and visualize sentiment distribution per cluster."""
    print("\nAnalyzing cluster x sentiment distribution...")
 
    sent_col = 'sentiment_label'
    sentiment_order  = ['positive', 'negative', 'neutral']
    sentiment_colors = {
        'positive': '#4daf4a',
        'negative': '#e41a1c',
        'neutral':  '#377eb8'
    }
    clusters = sorted(df['cluster'].unique())
 
    # ── Raw counts and percentages ────────────────────────────────────────────
    counts = (
        df.groupby(['cluster', sent_col])
        .size()
        .unstack(fill_value=0)
    )
    for s in sentiment_order:
        if s not in counts.columns:
            counts[s] = 0
    counts = counts[[s for s in sentiment_order if s in counts.columns]]
    counts['total'] = counts.sum(axis=1)
    pct = counts.drop(columns='total').div(counts['total'], axis=0) * 100
 
    print("\nCluster x Sentiment — Raw Counts:")
    print(counts.to_string())
    print("\nCluster x Sentiment — Row % (within cluster):")
    print(pct.round(2).to_string())
 
    # ── Save CSVs ─────────────────────────────────────────────────────────────
    counts_path = os.path.join(output_dir, f'{output_subdir}cluster_sentiment_counts.csv')
    pct_path    = os.path.join(output_dir, f'{output_subdir}cluster_sentiment_pct.csv')
    counts.to_csv(counts_path)
    pct.to_csv(pct_path)
    print(f"Saved: {counts_path}")
    print(f"Saved: {pct_path}")
 
    # ── Save JSON summary ─────────────────────────────────────────────────────
    summary = {}
    for c in clusters:
        cluster_df = df[df['cluster'] == c]
        row = {'total': int(len(cluster_df))}
        for s in sentiment_order:
            n = int((cluster_df[sent_col] == s).sum())
            row[s]          = n
            row[f'{s}_pct'] = round(n / len(cluster_df) * 100, 2) if len(cluster_df) > 0 else 0.0
        row['dominant_sentiment'] = cluster_df[sent_col].value_counts().idxmax()
        summary[f'cluster_{c}'] = row
 
    json_path = os.path.join(output_dir, f'{output_subdir}cluster_sentiment_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")
 
    # ── Plot: stacked % bar + grouped count bar ───────────────────────────────
    import matplotlib.ticker as mtick
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
 
    plot_colors = [sentiment_colors.get(s, '#999999') for s in pct.columns]
 
    pct.plot(kind='bar', stacked=True, ax=ax1, color=plot_colors, width=0.6)
    ax1.set_title('Sentiment Composition per Cluster', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Percentage (%)')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='Sentiment', bbox_to_anchor=(1.0, 1), loc='upper left')
 
    counts_plot = counts.drop(columns='total')
    x      = np.arange(len(clusters))
    n_sent = len(counts_plot.columns)
    width  = 0.8 / n_sent
 
    for i, s in enumerate(counts_plot.columns):
        offset = (i - n_sent / 2 + 0.5) * width
        bars = ax2.bar(
            x + offset,
            counts_plot[s],
            width=width,
            label=s,
            color=sentiment_colors.get(s, '#999999')
        )
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 0.5,
                    str(int(h)),
                    ha='center', va='bottom', fontsize=7
                )
 
    ax2.set_title('Sentiment Counts per Cluster', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Tweet Count')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Cluster {c}' for c in clusters], rotation=0)
    ax2.legend(title='Sentiment')
 
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'{output_subdir}cluster_sentiment_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")
 
    # ── Overall distribution ──────────────────────────────────────────────────
    overall = df[sent_col].value_counts()
    print(f"\nOverall sentiment distribution:")
    for s, n in overall.items():
        print(f"  {s:10s}: {n:6d}  ({n/len(df)*100:.1f}%)")
 
analyze_cluster_sentiment(df, cluster_labels, args.output_dir, output_subdir, args.n_topics, colors)
 
print(f"\n✓ Done. All results saved to {args.output_dir}/")