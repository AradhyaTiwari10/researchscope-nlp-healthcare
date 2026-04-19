"""
similarity.py
-------------
Module: Similarity Mapping
Phase 0 – ResearchScope NLP Healthcare

Responsibilities:
  - Compute cross-document similarity mathematically

Public API:
  compute_similarity(X: Any) -> Any
"""

from sklearn.metrics.pairwise import cosine_similarity
from typing import Any

def compute_similarity(X: Any) -> Any:
    """
    Compute cosine similarity between documents based on their feature matrix.

    Args:
        X (Any): The feature matrix (e.g., TF-IDF). 
                 Rows represent documents, columns represent features.

    Returns:
        Any: A dense similarity matrix of shape (n_samples, n_samples) 
             where the value at (i, j) is the cosine similarity between 
             document i and document j.
    """
    sim_matrix = cosine_similarity(X)
    return sim_matrix
