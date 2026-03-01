"""
Extractive Summarization Module.

This module provides a traditional, non-generative approach to summarizing 
research text by identifying and ranking sentences using TF-IDF importance.
Includes aggressive pre-summary cleaning to strip PDF artifacts.
"""

import re
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download("punkt")
    nltk.download("punkt_tab")

# Hard cap: never return more than 5 sentences regardless of caller's top_n
MAX_SENTENCES = 5

# Sentence-level blocklist: prefixes that indicate metadata/boilerplate lines
_BLOCKED_PREFIXES = (
    "send orders", "correspondence", "received", "accepted", "published",
    "copyright", "conflict", "declaration", "funding", "author contribution",
    "supplementary", "abbreviation", "keywords:", "abstract", "figure",
    "table", "reprints", "address correspondence",
)

def remove_references_section(text):
    """
    Splits on the References/Bibliography header and discards everything after it.
    This is the most impactful single cleaning step for scientific PDFs.
    """
    # Match standalone REFERENCES header (in all-caps, title case, or inline)
    parts = re.split(
        r'\n\s*(?:REFERENCES|References|Bibliography|Works Cited|BIBLIOGRAPHY)\s*\n',
        text,
        flags=re.IGNORECASE
    )
    return parts[0]

def fix_hyphenation(text):
    """
    Rejoins words broken by PDF line-wrap hyphenation.
    Handles both:
    Net-\ntive
    Net- tive
    """
    # Case 1: hyphen + newline
    text = re.sub(r"-\s*\n\s*", "", text)

    # Case 2: hyphen + space + lowercase continuation
    text = re.sub(r"-\s+(?=[a-z])", "", text)

    return text

def normalize_newlines(text):
    """
    Merges hard newlines mid-sentence. Scientific PDFs insert line breaks
    at column boundaries that break sentence tokenizers.
    Only merges when the next line starts with a lowercase letter
    (i.e. not a new sentence or section header).
    """
    # Newline followed by a lowercase letter → part of current sentence
    text = re.sub(r"\n(?=[a-z])", " ", text)
    return text

def remove_fragmented_lines(text):
    """
    Removes only clearly noisy lines (table rows, figure stubs, reference IDs).
    Uses a digit RATIO check so that real sentences with inline numbers are kept.
    """
    lines = text.split("\n")
    cleaned = []

    for line in lines:
        stripped = line.strip()

        # Drop truly empty or extremely short stubs (e.g. section labels, "Fig. 1")
        if len(stripped) < 10:
            continue

        # Drop lines where more than 40% of characters are digits
        # (table cells, pure reference rows) — but keep normal scientific sentences
        digit_count = len(re.findall(r"\d", stripped))
        if len(stripped) > 0 and digit_count / len(stripped) > 0.40:
            continue

        cleaned.append(stripped)

    return " ".join(cleaned)

def clean_for_summary(text):
    """
    Strips metadata, identifiers, and citation noise from research text
    to improve sentence ranking quality.
    NOTE: Assumes remove_references_section() and PDF normalization have
    already been applied upstream in extract_summary().
    """
    # 1. Remove article history blocks
    text = re.sub(r"ARTICLE HISTORY.*", "", text, flags=re.IGNORECASE)

    # 2. Remove pdfplumber CID font encoding artifacts: (cid:27), (cid:123) etc.
    text = re.sub(r"\(cid:\d+\)", "", text)

    # 3. Remove copyright / trademark lines
    text = re.sub(r"©.*?(rights reserved|publishers?|university|college|society)[^\n]*", "", text, flags=re.IGNORECASE)

    # 4. Remove emails
    text = re.sub(r"\S+@\S+", "", text)

    # 5. Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 6. Remove PDF filename artifacts and percent-encoded junk
    text = re.sub(r"\S+\.pdf\S*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"%\w+", "", text)

    # 4. Remove DOI identifiers (case-insensitive)
    text = re.sub(r"doi:\s*\S+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"10\.\d{4,}/\S+", "", text)  # Raw DOI format

    # 5. Remove PMID references
    text = re.sub(r"PMID:\s*\d+", "", text, flags=re.IGNORECASE)

    # 6. Remove inline citation brackets: [1], [12-15], [1,2,3]
    text = re.sub(r"\[\d[\d,\s\-]*\]", "", text)

    # 7. Remove (Author et al., YYYY) style citations
    text = re.sub(r"\(\w[\w\s]+et al\.,?\s*\d{4}\)", "", text)

    # NOTE: We intentionally do NOT strip standalone years (e.g. 2023).
    # Year stripping was found to destroy large portions of valid scientific prose.

    # 9. Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def _is_valid_sentence(s):
    """
    Returns True if the sentence is substantive narrative text.
    Rejects sentences that are too short, metadata-heavy, or boilerplate.
    """
    s_stripped = s.strip()
    s_lower = s_stripped.lower()
    words = s_stripped.split()

    # Reject all-caps section headers (e.g., "INTRODUCTION")
    if s_stripped.isupper():
        return False

    # Length filter — must have at least 8 words
    if len(words) < 8:
        return False

    # Reject lines that are mostly numeric (e.g. reference lists)
    digit_ratio = sum(c.isdigit() for c in s_stripped) / max(len(s_stripped), 1)
    if digit_ratio > 0.25:
        return False

    # Reject known boilerplate prefixes
    if any(s_lower.startswith(prefix) for prefix in _BLOCKED_PREFIXES):
        return False

    # Reject sentences that still contain DOI artifacts
    if "doi" in s_lower or "@" in s_stripped:
        return False

    return True

def extract_summary(text, top_n=5):
    """
    Extracts the top N informative sentences from a research document using
    length-normalized TF-IDF sentence scoring.

    Cleaning pipeline:
      1. Remove references section
      2. Fix PDF hyphenation artifacts
      3. Normalize mid-sentence newlines
      4. Strip fragmented table/figure lines
      5. Strip metadata identifiers (URLs, DOIs, emails, citations)
      6. Filter stub sentences
      7. Score and rank by length-normalized TF-IDF
    """
    # Hard cap
    top_n = min(top_n, MAX_SENTENCES)

    # Stage 1: References section removal (do this before any line processing)
    text = remove_references_section(text)

    # Stage 2: PDF-level normalization
    text = fix_hyphenation(text)
    text = normalize_newlines(text)
    text = remove_fragmented_lines(text)

    # Stage 3: Metadata / identifier cleaning
    cleaned_text = clean_for_summary(text)

    # Stage 2: Sentence tokenization
    all_sentences = sent_tokenize(cleaned_text)

    # Stage 3: Sentence-level quality filter
    sentences = [s for s in all_sentences if _is_valid_sentence(s)]

    if not sentences:
        return "Summary could not be generated: insufficient clean sentences extracted."

    if len(sentences) <= top_n:
        return " ".join(sentences)

    # Stage 4: TF-IDF sentence scoring (length-normalized)
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return " ".join(sentences[:top_n])

    # Length-normalized scoring: divide sum by sentence word count
    # This prevents long, metadata-dense sentences from dominating
    sentence_scores = np.array([
        tfidf_matrix.getrow(i).sum() / max(len(sentences[i].split()), 1)
        for i in range(len(sentences))
    ])

    # Stage 5: Pick top-N and restore original document order
    top_indices = sentence_scores.argsort()[::-1][:top_n]
    top_indices.sort()

    return " ".join([sentences[i] for i in top_indices])


if __name__ == "__main__":
    test_text = (
        "ARTICLE HISTORY Received 14 Jan 2023.\n"
        "author@university.ac.uk doi: 10.1001/ai.2023. PMID: 87654.\n"
        "Artificial Intelligence in healthcare has demonstrated significant "
        "improvements in early disease detection and clinical decision support. "
        "Machine learning models trained on large-scale electronic health records "
        "can predict patient deterioration hours before clinical symptoms manifest. "
        "Send Orders for Reprints to reprints@benthamopen.net. "
        "These approaches have been validated across multiple hospital settings "
        "with sensitivity values exceeding ninety percent. "
        "Deep neural networks applied to medical imaging have outperformed "
        "radiologists in detecting early-stage cancers. [1, 2]\n"
        "References\n"
        "1. LeCun et al. Deep Learning. Nature. 2015."
    )
    print("=== Summary ===")
    print(extract_summary(test_text, top_n=3))
