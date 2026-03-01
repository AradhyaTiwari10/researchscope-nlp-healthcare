"""
Text Preprocessing Pipeline Module.

This module provides a strictly traditional NLP pipeline for cleaning and
preparing research text for feature extraction and modeling.
Includes domain-specific stopword filtering and bigram detection.
"""

import re
import nltk
import spacy
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download("stopwords")

# Load NLTK stop words
_NLTK_STOPWORDS = set(stopwords.words("english"))

# Domain-specific stopwords for Healthcare/NLP research PDFs
# These are common PDF artifacts, figure labels, and overly-generic scientific terms
# that degrade topic coherence when included.
_DOMAIN_STOPWORDS = {
    # PDF artifacts
    "figure", "fig", "table", "et", "al", "crossref", "pmid", "doi",
    "copyright", "arxiv", "preprint", "researchgate",
    # Overly generic terms that appear in every paper
    "use", "used", "using", "datum", "data", "result", "results",
    "study", "studies", "paper", "method", "methods", "approach",
    "propose", "proposed", "show", "shown", "base", "based",
    "include", "including", "provide", "note", "section", "model",
    # Common abbreviations extracted as tokens
    "ieee", "acm", "vol", "pp", "http", "www", "org",
}

# Merged final stopword set
STOP_WORDS = _NLTK_STOPWORDS.union(_DOMAIN_STOPWORDS)

# Load spaCy model with essential components only
try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Run 'python -m spacy download en_core_web_sm'")
    nlp = None

def clean_text(text):
    """
    Performs basic cleaning: lowercasing, number removal, and punctuation removal.
    """
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_document(text):
    """
    Complete per-document pipeline: cleaning, tokenization, stopword removal,
    lemmatization, and length filtering.
    """
    if not nlp:
        return text.split()

    cleaned = clean_text(text)
    doc = nlp(cleaned)

    tokens = [
        token.lemma_
        for token in doc
        if token.lemma_ not in STOP_WORDS
        and len(token.lemma_) > 2
        and token.is_alpha
    ]

    return tokens

def preprocess_corpus(documents, detect_bigrams=True):
    """
    Applies the preprocessing pipeline to a collection of documents.
    Optionally runs gensim Phrases to surface bigrams (e.g., machine_learning).
    Returns a list of tokenized documents.
    """
    tokenized = [preprocess_document(doc) for doc in documents]

    if detect_bigrams and len(tokenized) > 1:
        # Train bigram model — min_count=2 means it must appear in ≥2 docs
        phrases_model = Phrases(tokenized, min_count=2, threshold=10)
        bigram = Phraser(phrases_model)
        tokenized = [bigram[doc] for doc in tokenized]

    return tokenized

if __name__ == "__main__":
    sample = "Machine learning and deep learning are used in figure 3 for healthcare diagnosis."
    print(f"Original: {sample}")
    tokens = preprocess_document(sample)
    print(f"Processed tokens: {tokens}")

    # Test that garbage tokens are absent
    bad_tokens = {"figure", "use", "datum", "et", "al"}
    found = bad_tokens.intersection(set(tokens))
    if found:
        print(f"⚠️  Garbage tokens still present: {found}")
    else:
        print("✅ No garbage tokens found.")