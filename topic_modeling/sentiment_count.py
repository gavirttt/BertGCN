import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create DataFrame
data = pd.DataFrame({
    'sentiment_label': ['negative', 'neutral', 'positive'],
    'count': [40799, 20155, 12845]
})

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

# Create bar plot
ax = sns.barplot(data=data, x='sentiment_label', y='count', 
                 palette=['#ff6b6b', '#4ecdc4', '#45b7d1'])

# Customize
plt.title('Sentiment Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Count', fontsize=12)

# Add value labels
for i, v in enumerate(data['count']):
    ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()