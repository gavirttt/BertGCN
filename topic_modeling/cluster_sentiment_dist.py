import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('topic_modeling/finetuned-model/3clusters/all-sentiment/all-months/cluster_sentiment_counts.csv')

# Prepare matrix
cluster_labels = [
    "Cluster 0",
    "Cluster 1",
    "Cluster 2"
]

# Rows = clusters, Columns = sentiments
data_matrix = df[['positive', 'negative', 'neutral']].copy()
data_matrix.index = cluster_labels  # Set cluster names as row index

# Create heatmap
plt.figure(figsize=(10, 6))
ax = sns.heatmap(
    data_matrix,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar_kws={'label': 'Number of Tweets', 'shrink': 0.8},
    linewidths=0.5,
    linecolor='white',
    square=False,  # Don't force square cells
    xticklabels=True,  # Show x-axis labels (positive, negative, neutral)
    yticklabels=True   # Show y-axis labels (clusters)
)

# Rotate y-axis labels to horizontal
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')

# Optional: Rotate x-axis labels if needed (keeps them horizontal)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

plt.title('Distribution of Tweets by Cluster and Sentiment', fontsize=14, fontweight='bold')
plt.ylabel('Topic Cluster', fontsize=12)
plt.xlabel('Sentiment', fontsize=12)

plt.tight_layout()
plt.savefig('topic_modeling/finetuned-model/3clusters/all-sentiment/all-months/cluster_sentiment_heatmap_seaborn.png', dpi=150)
plt.show()