"""
Text Preprocessing Pipeline Module.

This module provides a strictly traditional NLP pipeline for cleaning and
preparing research text for feature extraction and modeling. 
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

# Load stop words
STOP_WORDS = set(stopwords.words("english"))

# Load spaCy model with essential components only
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    # If not found, a note to user to download it
    print("Warning: spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

def clean_text(text):
    """
    Performs basic cleaning: lowercasing, number removal, and punctuation removal.
    """
    # Lowercasing
    text = text.lower()
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_document(text):
    """
    Complete per-document pipeline: cleaning, tokenization, stopword removal, 
    lemmatization, and length filtering.
    """
    if not nlp:
        return text.split() # Fallback if spacy is missing
        
    cleaned = clean_text(text)
    doc = nlp(cleaned)

    # Token-level processing
    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOP_WORDS
        and len(token.lemma_) > 2
        and token.is_alpha
    ]

    return tokens

def preprocess_corpus(documents):
    """
    Applies the preprocessing pipeline to a collection of documents.
    Returns a list of tokenized documents.
    """
    return [preprocess_document(doc) for doc in documents]

if __name__ == "__main__":
    # Sample verification
    sample = "Artificial Intelligence in Healthcare 2024: A systematic review."
    print(f"Original: {sample}")
    print(f"Processed: {preprocess_document(sample)}")