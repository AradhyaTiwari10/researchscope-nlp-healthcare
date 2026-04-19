"""
preprocessing.py
-----------------
Module: Text Preprocessing Pipeline
Phase 0 – ResearchScope NLP Healthcare

Responsibilities:
  - Ligature normalization (PDF artifacts)
  - Regex-based noise removal (citations, DOIs, URLs, OCR artifacts)
  - Tokenization and lemmatization (WordNet)
  - Stopword removal with domain-specific custom stopwords

Public API:
  preprocess(text: str) -> str
  clean_text(text: str) -> str
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# ── NLTK resource bootstrap ────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# ── Module-level singletons (initialised once) ─────────────────────────────────
_lemmatizer = WordNetLemmatizer()

# Domain-specific noise words that NLTK's generic stoplist misses for
# biomedical / ML-research documents.
_CUSTOM_STOPWORDS: set = {
    # Reference / citation artefacts
    "figure", "table", "et", "al", "pmid", "crossref",
    "doi", "http", "www", "com",
    # Over-represented generic research verbs
    "using", "based", "result", "study", "analysis", "method",
    "approach", "datum", "data", "number", "miss", "gain",
    # PDF layout tokens
    "drive", "note", "temperature", "surface", "magnetic", "sequence", "event",
    # Second-pass noise caught experimentally
    "present", "compare", "part", "show", "sect",
    "perform", "paper", "usually", "dataset", "need",
    "central", "start", "link", "layer", "group", "apply", "stress",
    # ML jargon that dominates topic-model output unhelpfully
    "machine", "learn", "learning", "algorithm",
    # Generic transition verbs
    "aim", "provide", "include", "discuss", "give", "discus",
    # Miscellaneous topic-model noise
    "ion", "property", "condition", "promising",
    "investigation", "type", "room",
    "used", "become", "since", "advent",
}

_STOP_WORDS: set = set(stopwords.words("english")).union(_CUSTOM_STOPWORDS)

# Unicode ligatures that commonly appear in PDF-extracted text
_LIGATURE_MAP: dict = {
    "\ufb01": "fi",  # ﬁ
    "\ufb02": "fl",  # ﬂ
    "\ufb00": "ff",  # ﬀ
    "\ufb03": "ffi", # ﬃ
    "\ufb04": "ffl", # ﬄ
    "\ufb05": "st",  # ﬅ
    "\ufb06": "st",  # ﬆ
    "\u00c6": "AE",  # Æ
    "\u00e6": "ae",  # æ
    "\u0152": "OE",  # Œ
    "\u0153": "oe",  # œ
}


# ── Private helpers ────────────────────────────────────────────────────────────

def _fix_ligatures(text: str) -> str:
    """Replace Unicode ligatures with their ASCII equivalents."""
    for ligature, replacement in _LIGATURE_MAP.items():
        text = text.replace(ligature, replacement)
    return text


def clean_text(text: str) -> str:
    """
    Apply all regex-based cleaning steps to raw text.

    Steps (order matters):
      1. Ligature normalisation
      2. Hyphenation repair  (e.g. "technol-\\nogy" → "technology")
      3. ALL-CAPS metadata headers removal
      4. OCR / PDF glyph artefacts  (/g415, /e190, …)
      5. Legal symbols (©, ®, ™)
      6. Emails, URLs, DOIs
      7. Numeric citations  [15], (15,16), …
      8. Standalone numbers
      9. Domain-irrelevant chemical/physics vocabulary
      10. Very short tokens (≤2 chars) – removes many meaningless acronyms
      11. Orphan dashes left by short-token removal
      12. Empty parentheses  ( )
      13. Whitespace normalisation
    """
    text = _fix_ligatures(text)

    # 1. Repair soft-hyphenated words split across lines
    text = re.sub(r'([A-Za-z]+)-\s+([A-Za-z]+)', r'\1\2', text)

    # 2. Remove ALL-CAPS header blocks (e.g. "REVIEW ARTICLE", "HISTORY Received:")
    text = re.sub(r'\b[A-Z][A-Z\s]+[A-Z]\b:?', " ", text)

    # 3. OCR / PDF glyph artefacts like /g415, /e190
    text = re.sub(r'/[a-z]\d+', " ", text, flags=re.IGNORECASE)

    # 4. Legal symbols
    text = re.sub(r'[©®™]', " ", text)

    # 5. Emails
    text = re.sub(r"\S+@\S+", " ", text)

    # 6. URLs and DOIs
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"doi\S+", " ", text, flags=re.IGNORECASE)

    # 7. Numeric citations: [15], [1,2,3], (15), (1, 2)
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    text = re.sub(r"\(\d+(?:\s*,\s*\d+)*\)", " ", text)

    # 8. Standalone numbers
    text = re.sub(r"\b\d+\b", " ", text)

    # 9. Domain-irrelevant vocabulary (biomedical / physics terms)
    _irrelevant = r"\b(?:bond|material|vibration|oxide|acid|crystal|temperature|surface|magnetic)\b"
    text = re.sub(_irrelevant, " ", text, flags=re.IGNORECASE)

    # 10. Remove 1–2 letter tokens
    text = re.sub(r"\b[A-Za-z]{1,2}\b", " ", text)

    # 11. Orphan dashes produced by step 10
    text = re.sub(r"\s+-\b", " ", text)   # " -word" → " word"
    text = re.sub(r"\b-\s+", " ", text)   # "word- " → "word "

    # 12. Empty parentheses
    text = re.sub(r"\(\s*\)", " ", text)

    # 13. Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Public API ─────────────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline for a single document string.

    Pipeline:
      clean_text → lowercase tokenisation → lemmatisation →
      stopword + punctuation + short-token filtering → rejoin

    Args:
        text: Raw text extracted from a PDF or any source.

    Returns:
        A single whitespace-joined string of cleaned, lemmatized tokens.
        Returns an empty string if the input is empty/whitespace.

    Example:
        >>> from src.preprocessing import preprocess
        >>> preprocess("The patient's EHR data showed significant ﬁndings.")
        'patient ehr show significant finding'
    """
    if not text or not text.strip():
        return ""

    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned.lower())

    processed_tokens = [
        _lemmatizer.lemmatize(word)
        for word in tokens
        if word not in _STOP_WORDS          # remove stopwords
        and word not in string.punctuation  # remove punctuation marks
        and word.isalpha()                  # keep only alphabetic tokens
        and len(word) > 3                   # drop very short tokens
    ]

    return " ".join(processed_tokens)