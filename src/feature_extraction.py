"""
TF-IDF Feature Extraction Module.

This module uses scikit-learn to transform preprocessed text data into 
numerical features and extract significant keywords from documents.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(processed_docs, max_features=1000):
    """
    Fits TfidfVectorizer on preprocessed documents (list of lists of tokens).
    Returns the vectorizer and the TF-IDF matrix.
    """
    # Join tokens back into strings for TfidfVectorizer
    raw_text = [" ".join(doc) for doc in processed_docs]
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(raw_text)
    
    return vectorizer, tfidf_matrix

def get_top_keywords_per_doc(vectorizer, tfidf_matrix, top_n=20):
    """
    Extracts the top N terms for each document based on TF-IDF scores.
    """
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_keywords = []
    
    for i in range(tfidf_matrix.shape[0]):
        # Get scores for the current document
        row = tfidf_matrix.getrow(i).toarray().flatten()
        # Sort indices by score in descending order
        top_indices = row.argsort()[::-1][:top_n]
        # Map indices to terms
        keywords = feature_names[top_indices]
        # Filter out terms with zero score (if any)
        valid_keywords = [keywords[j] for j in range(len(top_indices)) if row[top_indices[j]] > 0]
        top_keywords.append(valid_keywords)
        
    return top_keywords

def get_global_top_terms(vectorizer, tfidf_matrix, top_n=20):
    """
    Calculates global top terms based on mean TF-IDF score across the corpus.
    """
    feature_names = vectorizer.get_feature_names_out()
    mean_tfidf = tfidf_matrix.mean(axis=0).A1
    top_indices = mean_tfidf.argsort()[::-1][:top_n]
    
    return [(feature_names[i], mean_tfidf[i]) for i in top_indices]

if __name__ == "__main__":
    # Sample verification
    sample_docs = [
        ["heart", "disease", "ai", "diagnosis", "healthcare"],
        ["machine", "learning", "clinical", "data", "patient", "care"],
        ["heart", "rate", "monitoring", "wearable", "ai"]
    ]
    vec, matrix = extract_tfidf_features(sample_docs)
    print("Top Keywords per Document:")
    print(get_top_keywords_per_doc(vec, matrix, top_n=3))
    print("\nGlobal Top Terms:")
    print(get_global_top_terms(vec, matrix, top_n=5))
