import pandas as pd
import os
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

def compute_global_word_sentiment_distribution(all_words, texts_cleaned, df):
    """
    Compute sentiment distribution for each word based on all tweets it appears in,
    globally across the entire filtered dataset.
    """
    global_word_sentiment_data = {}
    
    for word in all_words:
        sentiment_counts = {
            'positive': 0,
            'negative': 0,
            'neutral': 0,
            'total': 0
        }
        
        # Find all tweets containing this word in the entire filtered dataset
        for idx, text in enumerate(texts_cleaned):
            if word.lower() in text.lower():
                sentiment = df.iloc[idx]['sentiment_label']
                sentiment_counts[sentiment] += 1
                sentiment_counts['total'] += 1
        
        if sentiment_counts['total'] > 0:
            proportions = {
                'positive': sentiment_counts['positive'] / sentiment_counts['total'],
                'negative': sentiment_counts['negative'] / sentiment_counts['total'],
                'neutral': sentiment_counts['neutral'] / sentiment_counts['total']
            }
            dominant = max(proportions, key=proportions.get)
            confidence = proportions[dominant]
        else:
            proportions = {'positive': 0, 'negative': 0, 'neutral': 0}
            dominant = 'neutral' # Default if word not found anywhere
            confidence = 0.5
        
        global_word_sentiment_data[word] = {
            'counts': sentiment_counts,
            'proportions': proportions,
            'dominant': dominant,
            'confidence': confidence
        }
    
    return global_word_sentiment_data

def sentiment_color_func_global(word, global_word_sentiment_data, font_size=None, 
                                 position=None, orientation=None, font_path=None, 
                                 random_state=None):
    """
    Color function for word cloud based on globally determined sentiment distribution.
    """
    sentiment_colors = {
        'positive': '#4daf4a',
        'negative': '#e41a1c',
        'neutral':  "#777777" # Changed neutral to a distinct color
    }

    if word not in global_word_sentiment_data:
        return '#000000' # Default to black if word not found in global data

    dominant_sentiment = global_word_sentiment_data[word]['dominant']
    return sentiment_colors[dominant_sentiment]

def regenerate_global_sentiment_wordclouds(
    input_dir,
    output_parent_dir=None,
    n_topics=5
):
    print(f"\n--- Regenerating Global Sentiment Word Clouds ---")
    
    # Determine output directory
    if output_parent_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_parent_dir = os.path.join(input_dir, f'global_sentiment_wcs_{timestamp}')
    
    sentiment_wc_dir = os.path.join(output_parent_dir, 'sentiment_wordclouds_global')
    os.makedirs(sentiment_wc_dir, exist_ok=True)
    print(f"Output directory for global word clouds: {sentiment_wc_dir}")

    # Load data
    try:
        clusters_csv_path = os.path.join(input_dir, 'tweets_with_clusters.csv')
        df = pd.read_csv(clusters_csv_path)
        print(f"Loaded {len(df)} tweets from {clusters_csv_path}")

        keywords_json_path = os.path.join(input_dir, 'llr_keywords.json')
        with open(keywords_json_path, 'r', encoding='utf-8') as f:
            llr_results = json.load(f)
        print(f"Loaded LLR keywords from {keywords_json_path}")

    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please ensure '{e.filename}' exists in '{input_dir}'")
        print("The script expects 'tweets_with_clusters.csv' and 'llr_keywords.json' ")
        print("to be present in the specified input directory.")
        return
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return

    # Prepare texts_cleaned (same logic as topic_modeling.py)
    texts_cleaned = df['cleaned_text2'].fillna('').tolist() \
        if 'cleaned_text2' in df.columns \
        else df['text'].fillna('').tolist()
    
    if not texts_cleaned:
        print("No text data found for analysis. Exiting.")
        return

    # Collect all unique keywords across all clusters to compute global sentiment
    all_unique_keywords = set()
    for cluster_key in llr_results:
        for word, _ in llr_results[cluster_key]:
            all_unique_keywords.add(word)
    
    print(f"Computing global sentiment for {len(all_unique_keywords)} unique keywords...")
    global_word_sentiment_data = compute_global_word_sentiment_distribution(
        list(all_unique_keywords), texts_cleaned, df
    )
    print("Global sentiment distribution computed.")

    # Generate word clouds for each cluster using global sentiment coloring
    WC_MAX_WORDS = 50
    for c in range(n_topics):
        print(f"\nGenerating global sentiment word cloud for Cluster {c}...")
        
        keywords = llr_results[f'cluster_{c}']
        word_weights = {w: s for w, s in keywords[:WC_MAX_WORDS] if s > 0}
        
        if not word_weights:
            print(f"  Cluster {c}: no positive LLR scores, skipping.")
            continue
        
        def color_func(word, *args, wsd=global_word_sentiment_data, **kwargs):
            return sentiment_color_func_global(word, wsd)
        
        wc = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            color_func=color_func,
            max_words=WC_MAX_WORDS,
            prefer_horizontal=0.9,
            random_state=42,
            collocations=False
        )
        wc.generate_from_frequencies(word_weights)
        
        path = os.path.join(sentiment_wc_dir, f'cluster{c}_global_sentiment_wordcloud.png')
        wc.to_file(path)
        print(f"  Global sentiment word cloud for Cluster {c} saved to {path}")

    # Create a combined grid of global sentiment-colored word clouds
    cols = 2
    rows = (n_topics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 6 * rows))
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for c in range(n_topics):
        keywords = llr_results[f'cluster_{c}']
        word_weights = {w: s for w, s in keywords[:WC_MAX_WORDS] if s > 0}
        
        if word_weights:
            def color_func(word, *args, wsd=global_word_sentiment_data, **kwargs):
                return sentiment_color_func_global(word, wsd)
            
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

        axes[c].set_title(f'Cluster {c}', fontsize=13, fontweight='bold')
        axes[c].axis('off')
    
    # Hide empty subplots
    for c in range(n_topics, len(axes)):
        axes[c].axis('off')
    
    plt.suptitle(f'Topic Word Clouds with GLOBAL Sentiment Coloring\n(Based on Tweet-Level Sentiment Labels)', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    combined_path = os.path.join(sentiment_wc_dir, 'all_clusters_global_sentiment_wordclouds.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nCombined global sentiment word cloud grid saved to {combined_path}")

    print(f"\n✓ Done. Global sentiment word clouds regenerated and saved to {sentiment_wc_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing tweets_with_clusters.csv and llr_keywords.json from a previous run.')
    parser.add_argument('--output_parent_dir', type=str, default=None,
                        help='Optional: Parent directory to save the new word clouds. If not specified, ')
    parser.add_argument('--n_topics', type=int, default=5,
                        help='Number of topics used in the original run.')
    args = parser.parse_args()

    regenerate_global_sentiment_wordclouds(
        args.input_dir,
        args.output_parent_dir,
        args.n_topics
    )

## python regenerate_wordclouds.py --n_topics 3 --input_dir finetuned-model\3clusters\all-sentiment\all-months --output_parent_dir finetuned-model\3clusters\all-sentiment\all-months\global-sentiment_wordclouds
## python regenerate_wordclouds.py --n_topics 4 --input_dir finetuned-model\4clusters\all-sentiment\all-months --output_parent_dir finetuned-model\4clusters\all-sentiment\all-months\global-sentiment_wordclouds
## python regenerate_wordclouds.py --n_topics 5 --input_dir finetuned-model\5clusters\all-sentiment\all-months --output_parent_dir finetuned-model\5clusters\all-sentiment\all-months\global-sentiment_wordclouds