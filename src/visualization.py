from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud)
    ax.axis("off")
    
    return fig

def generate_cosine_heatmap(similarity_matrix, file_names):
    df = pd.DataFrame(similarity_matrix, index=file_names, columns=file_names)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="YlGnBu", ax=ax, fmt=".2f")
    ax.set_title("Cross-Document Cosine Similarity Heatmap")
    
    return fig