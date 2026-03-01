"""
Extractive Summarization Module.

This module provides a traditional, non-generative approach to summarizing 
research text by identifying and ranking sentences using TF-IDF importance. 
"""

import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

def extract_summary(text, top_n=5):
    """
    Ranks sentences by their TF-IDF scores within the current document text 
    and returns a summary of the top N sentences.
    """
    # 1. Tokenize document into sentences
    sentences = sent_tokenize(text)
    
    if len(sentences) <= top_n:
        return text 

    # 2. Vectorize sentences as a pseudo-corpus of the document
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Sum TF-IDF scores for each sentence (vector)
        # Score = Average of non-zero TF-IDF values in sentence
        sentence_scores = tfidf_matrix.sum(axis=1).A1
    except ValueError:
        # Happens if sentences are too short or only contain stop words
        return " ".join(sentences[:top_n])
    
    # 3. Identify and sort indices of top-scoring sentences
    top_indices = sentence_scores.argsort()[::-1][:top_n]
    
    # Keep sentences in their original order for readability
    top_indices.sort()
    
    # 4. Construct the summary
    return " ".join([sentences[i] for i in top_indices])

if __name__ == "__main__":
    # Internal test
    test_text = (
        "Artificial Intelligence is a powerful technology. It is being used across multiple "
        "fields, including healthcare and finance. In healthcare, it can help in predicting "
        "diseases early and accurately. This improves patient care significantly. Researchers "
        "around the world are investing heavily into this field to improve its capabilities. "
        "The model uses traditional NLP for this specific implementation. We avoid LLMs."
    )
    summary = extract_summary(test_text, top_n=3)
    print(f"Summary: {summary}")
