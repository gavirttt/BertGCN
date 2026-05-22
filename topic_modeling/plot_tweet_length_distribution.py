import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def plot_tweet_length_distribution(csv_file_path, output_dir=".", output_prefix="tweet_length_distribution"):
    """
    Generates separate figures for each sentiment label showing tweet length distribution.
    Saves each sentiment's plot as an individual image.

    Args:
        csv_file_path (str): The path to the input CSV file.
        output_dir (str): The directory to save the output image files.
        output_prefix (str): Prefix for the output image filenames.
    """
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        return
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'text' not in df.columns:
        print("Error: 'text' column not found in the CSV file.")
        return
    if 'sentiment' not in df.columns:
        print("Error: 'sentiment' column not found in the CSV file.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set global font sizes
    plt.rcParams.update({
        'font.size': 16,           # General font size
        'axes.titlesize': 20,      # Title font size
        'axes.labelsize': 18,      # X and Y label font size
        'xtick.labelsize': 14,     # X-tick labels font size
        'ytick.labelsize': 14,     # Y-tick labels font size
        'legend.fontsize': 16,     # Legend font size
        'legend.title_fontsize': 17  # Legend title font size
    })

    df['tweet_length'] = df['text'].apply(len)
    unique_sentiments = sorted(df['sentiment'].unique())

    # Define colors for different sentiments
    sentiment_colors = {
        'negative': 'salmon',
        'neutral': 'lightblue', 
        'positive': 'lightgreen'
    }
    default_color = 'steelblue'

    saved_files = []

    for sentiment in unique_sentiments:
        subset_df = df[df['sentiment'] == sentiment]
        
        # Create a single plot for this sentiment
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Choose color based on sentiment or use default
        color = sentiment_colors.get(sentiment.lower(), default_color)
        
        # Create histogram with KDE
        sns.histplot(data=subset_df, x='tweet_length', kde=True, color=color, alpha=0.6, ax=ax)
        
        # Calculate statistics
        mean_len = subset_df['tweet_length'].mean()
        median_len = subset_df['tweet_length'].median()
        max_len = subset_df['tweet_length'].max()
        n_count = len(subset_df)
        
        # Add vertical lines for mean and median (font size increased via rcParams)
        ax.axvline(mean_len, color='red', linestyle='--', alpha=0.7, linewidth=2, label=f'Mean: {mean_len:.1f}')
        ax.axvline(median_len, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'Median: {median_len:.1f}')
        
        # Set title and labels (sizes now controlled by rcParams)
        ax.set_title(f'{sentiment.capitalize()} Sentiment', fontweight='bold')
        ax.set_xlabel('Tweet Length (characters)')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', alpha=0.75)
        ax.legend(loc='upper right')
        
        # Add text box with summary statistics - INCREASED FONT SIZE HERE
        stats_text = f'Total Posts: {n_count:,}\nMax Length: {max_len:,} chars'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='gray'),
                fontsize=14,  # Changed from 10 to 14
                family='monospace')
        
        plt.tight_layout()
        
        # Save the figure
        output_filename = f"{output_prefix}_{sentiment}.png"
        output_path = os.path.join(output_dir, output_filename)
        
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved successfully to {output_path}")
            saved_files.append(output_path)
        except Exception as e:
            print(f"Error saving figure for {sentiment}: {e}")
        
        plt.close()

    print(f"\nSummary: {len(saved_files)} images saved to {output_dir}/")
    return saved_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate separate tweet length distribution plots for each sentiment.')
    parser.add_argument('csv_file_path', type=str, help='Path to the input CSV file.')
    parser.add_argument('--output-dir', type=str, default='.', help='Directory to save output image files.')
    parser.add_argument('--prefix', type=str, default='tweet_length_distribution', 
                        help='Prefix for output image filenames (e.g., "myplot" creates myplot_negative.png, etc.)')
    
    args = parser.parse_args()
    plot_tweet_length_distribution(args.csv_file_path, args.output_dir, args.prefix)