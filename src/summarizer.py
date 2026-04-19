"""
summarizer.py
--------------
Module: Text Summarization
Phase 0 – ResearchScope NLP Healthcare

Responsibilities:
  - Extractive summarization using TF-IDF ranking

Public API:
  summarize(text: str, vectorizer: TfidfVectorizer, top_n: int = 3) -> str
"""

import nltk
import re
from nltk.tokenize import sent_tokenize
from typing import Any
from src.preprocessing import clean_text

# Download required NLTK resources silently
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def _get_abstract_only(text: str) -> str:
    """
    Extract only the Abstract part of a research paper, if identifiable.
    """
    # Start at Abstract if present
    abstract_match = re.search(r'(?i)(?<![a-zA-Z])abstract\s*[:\n]?', text)
    if abstract_match:
        text = text[abstract_match.end():]
        
    # End at Introduction or Background
    intro_match = re.search(r'\n\s*(?:[IVX0-9]+\.?\s*)?(?:Introduction|Background)\b', text, re.IGNORECASE)
    if intro_match:
        text = text[:intro_match.start()]
        
    return text.strip()

def summarize(text: str, vectorizer: Any, top_n: int = 3) -> str:
    """
    Generate an extractive summary of the text by scoring sentences with a TF-IDF vectorizer.
    
    Args:
        text (str): The raw document text to summarize.
        vectorizer (Any): An initialized feature vectorizer instance (e.g. TfidfVectorizer).
                          A new instance can be provided, but it will be fit specifically 
                          to the sentences of this document.
        top_n (int): Number of sentences to include in the compiled summary.
        
    Returns:
        str: A concatenated string of the highest-scoring sentences, preserving their original order.
    """
    # 1. Isolate abstract or first portion for summarization focus
    abstract_text = _get_abstract_only(text)
    if not abstract_text:
        abstract_text = text[:2000] # Fallback to first 2000 characters if structure is unclear

    # 2. Tokenize original sentences first to preserve punctuation and readability
    original_sentences = sent_tokenize(abstract_text)
    if len(original_sentences) <= top_n:
        return " ".join(original_sentences)
        
    # 3. Clean sentences for feature extraction scoring ONLY
    cleaned_sentences = [clean_text(s) for s in original_sentences]
    
    # 4. Feature Extraction
    # We fit the provided vectorizer specifically on these cleaned sentences in order to score them
    try:
        X = vectorizer.fit_transform(cleaned_sentences)
    except ValueError:
        # Happens if vocab is empty or sentences are too short
        return " ".join(original_sentences[:top_n])
    
    # 5. Score Sentences
    sentence_scores = X.sum(axis=1) # Sum of TF-IDF scores across words in the sentence
    
    # 6. Rank Sentences
    ranked = sorted(
        [(sentence_scores[i, 0], i) for i in range(len(original_sentences))],
        reverse=True
    )
    
    # 7. Reconstruct Summary (maintaining original order)
    top_indices = sorted([idx for score, idx in ranked[:top_n]])
    summary = " ".join([original_sentences[idx] for idx in top_indices])
    
    return summary