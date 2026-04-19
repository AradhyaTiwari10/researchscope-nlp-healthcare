"""
topic_modeling.py
------------------
Module: Topic Modeling
Phase 0 – ResearchScope NLP Healthcare

Responsibilities:
  - Discover latent topics in numerical features (TF-IDF)
  - Latent Dirichlet Allocation (LDA)

Public API:
  get_topics(X: Any, vectorizer: TfidfVectorizer, num_topics: int = 4, num_words: int = 10) -> list[tuple]
"""

from sklearn.decomposition import LatentDirichletAllocation
from typing import Any, List, Tuple

def get_topics(X: Any, vectorizer: Any, num_topics: int = 4, num_words: int = 10) -> List[Tuple[int, List[str]]]:
    """
    Perform Latent Dirichlet Allocation (LDA) to extract topics from a feature matrix.

    Args:
        X (Any): The TF-IDF sparse matrix (or any feature matrix).
        vectorizer (Any): The fitted vectorizer used to generate X (must support get_feature_names_out).
        num_topics (int): The number of topics to extract.
        num_words (int): The number of top words to return per topic.

    Returns:
        List[Tuple[int, List[str]]]: A list of topics, where each topic is a tuple 
                                     containing the topic index and a list of its top words.
    """
    # Fit LDA model
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(X)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-num_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append((topic_idx, top_words))
        
    return topics