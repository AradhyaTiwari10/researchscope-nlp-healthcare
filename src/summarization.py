"""
Extractive Summarization Module.

This module provides a traditional, non-generative approach to summarizing 
research text by identifying and ranking sentences using TF-IDF importance. 
"""

import re
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

def clean_for_summary(text):
    """
    Cleans research text by removing noise like URLs, DOIs, citation blocks, 
    and the references section to improve sentence ranking.
    """
    # 1. Remove References/Bibliography section (usually at the end)
    # Look for common headers and split
    parts = re.split(r'\n\s*(?:References|Bibliography|Works Cited)\s*\n', text, flags=re.IGNORECASE)
    text = parts[0]
    
    # 2. Remove URLs
    text = re.sub(r"http\S+", "", text)
    
    # 3. Remove identifiers (PMID, DOI)
    text = re.sub(r"PMID:\s*\d+", "", text)
    text = re.sub(r"doi:\s*\S+", "", text)
    
    # 4. Remove Emails
    text = re.sub(r"\S+@\S+", "", text)
    
    # 5. Remove Citation blocks (e.g., [1, 2], [10-15], (Author et al., 2020))
    text = re.sub(r"\[[\d,\s-]+\]", "", text)
    text = re.sub(r"\(\w+\s+et\s+al\.,\s+\d{4}\)", "", text)
    
    # 6. Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def extract_summary(text, top_n=5):
    """
    Ranks sentences by their TF-IDF scores within the current document text 
    and returns a summary of the top N sentences.
    """
    # Pre-cleaning to remove non-narrative noise
    cleaned_text = clean_for_summary(text)
    
    # 1. Tokenize document into sentences
    sentences = sent_tokenize(cleaned_text)
    
    if len(sentences) <= top_n:
        return cleaned_text 

    # 2. Vectorize sentences as a pseudo-corpus of the document
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
        # Sum TF-IDF scores for each sentence (vector)
        sentence_scores = tfidf_matrix.sum(axis=1).A1
    except ValueError:
        # Happens if sentences are too short or only contain stop words after cleaning
        return " ".join(sentences[:top_n])
    
    # 3. Identify and sort indices of top-scoring sentences
    top_indices = sentence_scores.argsort()[::-1][:top_n]
    
    # Keep sentences in their original order for readability
    top_indices.sort()
    
    # 4. Construct the summary
    return " ".join([sentences[i] for i in top_indices])

if __name__ == "__main__":
    # Internal test with noise
    test_text = (
        "AI diagnosis is helpful [1, 2]. Visit http://example.com for more. "
        "The model showed 95% accuracy (Jones et al., 2021). "
        "PMID: 12345678 doi: 10.1001/ai.2023. "
        "This is a key finding in healthcare. "
        "References "
        "1. Paper one. 2. Paper two."
    )
    summary = extract_summary(test_text, top_n=2)
    print(f"Summary: {summary}")
