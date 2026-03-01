import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

lemmatizer = WordNetLemmatizer()
custom_stopwords = {
    "figure", "table", "et", "al", "pmid", "crossref",
    "doi", "http", "www", "com",
    "using", "based", "result", "study",
    "analysis", "method", "approach",
    "datum", "data", "number", "miss", "gain",
    "drive", "note", "temperature", "surface", 
    "magnetic", "sequence", "event",
    # Additional noise words from second iteration:
    "present", "compare", "part", "show", "sect",
    "perform", "paper", "usually", "dataset", "need",
    "central", "start", "link", "layer", "group", "apply", "stress",
    "machine", "learn", "learning", "algorithm" # We know it's about ML, so these can dominate unhelpfully
}

stop_words = set(stopwords.words("english")).union(custom_stopwords)


def clean_text(text):
    # 1. Fix Hyphenation (words split across lines or spaces like "technol- ogy" or "technol-\nogy")
    text = re.sub(r'([A-Za-z]+)-\s+([A-Za-z]+)', r'\1\2', text)
    
    # 2. Remove ALL CAPS headers/metadata (like "REVIEW ARTICLE", "HISTORY Received:")
    text = re.sub(r'\b[A-Z][A-Z\s]+[A-Z]\b:?', " ", text)
    
    # Remove emails
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove URLs and DOIs
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"doi\S+", " ", text, flags=re.IGNORECASE)
    
    # Remove citation numbers [15], (15), etc.
    text = re.sub(r"\[\d+(?:\s*,\s*\d+)*\]", " ", text)
    text = re.sub(r"\(\d+(?:\s*,\s*\d+)*\)", " ", text)
    
    # Remove numbers
    text = re.sub(r"\b\d+\b", " ", text)
    
    # Remove biomedical chemical short forms / irrelevant vocabulary
    irrelevant_vocab = r"\b(?:bond|material|vibration|oxide|acid|crystal|temperature|surface|magnetic)\b"
    text = re.sub(irrelevant_vocab, " ", text, flags=re.IGNORECASE)
    
    # remove 1-2 letter words (this strips acronyms like 'AI')
    text = re.sub(r"\b[A-Za-z]{1,2}\b", " ", text)
    
    # 3. Clean up empty parentheses left behind when acronyms are stripped (e.g. " ( ) ")
    text = re.sub(r"\(\s*\)", " ", text)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text):
    text = clean_text(text)

    tokens = word_tokenize(text.lower())

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
        and word not in string.punctuation
        and word.isalpha()
        and len(word) > 3
    ]

    return " ".join(tokens)