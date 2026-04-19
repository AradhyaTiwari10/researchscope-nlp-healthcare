"""
feature_engineering.py
-----------------------
Module: Feature Engineering
Phase 0 – ResearchScope NLP Healthcare

Responsibilities:
  - Convert preprocessed textual data into numerical features
  - TF-IDF vectorization

Public API:
  extract_features(texts: list[str], max_features: int = 2500, ngram_range: tuple = (1, 2)) -> tuple
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Tuple, Any

def extract_features(texts: List[str], max_features: int = 2500, ngram_range: tuple = (1, 2)) -> Tuple[Any, TfidfVectorizer]:
    """
    Extract TF-IDF features from a list of preprocessed documents.

    Args:
        texts (List[str]): A list of preprocessed string documents.
        max_features (int): Maximum number of features to extract.
        ngram_range (tuple): Range of n-grams to extract.

    Returns:
        Tuple[Any, TfidfVectorizer]: A tuple containing the sparse feature matrix (X) 
                                     and the fitted TfidfVectorizer object.
    
    Example:
        >>> X, vectorizer = extract_features(["machine learning model", "deep learning healthcare"])
    """
    if not texts:
        raise ValueError("Input texts list cannot be empty.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=2,              # Ignore terms that appear in less than 2 documents
        max_df=0.85,           # Ignore terms that appear in more than 85% of documents
        ngram_range=ngram_range
    )
    
    # In case we only have one document (e.g. testing), we should bypass min_df
    if len(texts) == 1:
        vectorizer.min_df = 1
        
    X = vectorizer.fit_transform(texts)
    return X, vectorizer