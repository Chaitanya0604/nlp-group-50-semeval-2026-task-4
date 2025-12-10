import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import numpy as np
import re

def analyze_themes(file_path, n_topics=5, n_top_words=10):
    print(f"Loading data from {file_path}...")
    # Load data (handling potential JSONL format or CSV based on extension)
    df = pd.read_json(file_path, lines=True)

    text_data = df['anchor_text'].dropna().astype(str).tolist()

    # Preprocessing
    print("Preprocessing text...")
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        return text

    text_data = [clean_text(t) for t in text_data]

    # Vectorization
    print("Vectorizing text...")
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tf = tf_vectorizer.fit_transform(text_data)

    # LDA Model
    print(f"Fitting LDA model with {n_topics} topics...")
    lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='online', random_state=42)
    lda.fit(tf)

    print("\n" + "="*40)
    print(f"Top {n_topics} Themes in Training Data")
    print("="*40)

    feature_names = tf_vectorizer.get_feature_names_out()
    
    # Prepare data for visualization
    theme_words = []
    for topic_idx, topic in enumerate(lda.components_):
        message = f"Theme {topic_idx + 1}: "
        top_words = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        message += ", ".join(top_words)
        print(message)
        theme_words.append((topic_idx + 1, top_words, topic[topic.argsort()[:-n_top_words - 1:-1]]))
    print("="*40)
    
    # Create visualization
    print("\nGenerating visualization...")
    fig, axes = plt.subplots(n_topics, 1, figsize=(12, 3*n_topics))
    if n_topics == 1:
        axes = [axes]
    
    for idx, (theme_num, words, weights) in enumerate(theme_words):
        ax = axes[idx]
        y_pos = np.arange(len(words))
        ax.barh(y_pos, weights[::-1], color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words[::-1])
        ax.set_xlabel('Weight')
        ax.set_title(f'Theme {theme_num}: Top {n_top_words} Words', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = 'themes_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    DATA_FILE = "data/SemEval2026-Task_4-dev-v1/dev_track_a.jsonl" 
    
    analyze_themes(DATA_FILE)
