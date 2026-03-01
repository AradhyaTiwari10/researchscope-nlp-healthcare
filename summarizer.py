import nltk
import re
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessing import clean_text

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

def get_abstract_only(text):
    # Start at Abstract if present, handling cases where it's glued to numbers like 124359Abstract:
    # (?<![a-zA-Z]) ensures it is not part of a larger word
    abstract_match = re.search(r'(?i)(?<![a-zA-Z])abstract\s*[:\n]?', text)
    if abstract_match:
        text = text[abstract_match.end():]
        
    # End at Introduction or Background
    intro_match = re.search(r'\n\s*(?:[IVX0-9]+\.?\s*)?(?:Introduction|Background)\b', text, re.IGNORECASE)
    if intro_match:
        text = text[:intro_match.start()]
        
    return text.strip()

def extractive_summary(text, num_sentences=3):
    # 1. Isolate abstract
    abstract_text = get_abstract_only(text)
    if not abstract_text:
        abstract_text = text[:2000] # Fallback to first 2000 chars

    # 2. Remove metadata
    clean_abstract = clean_text(abstract_text)
    
    sentences = sent_tokenize(clean_abstract)
    if len(sentences) <= num_sentences:
        return clean_abstract
        
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    sentence_scores = X.sum(axis=1)
    
    # Map sentences back to their original index
    ranked = sorted(
        [(sentence_scores[i, 0], i)
         for i in range(len(sentences))],
        reverse=True
    )
    
    # Get the indices of the top sentences and sort them to maintain original order
    top_indices = sorted([idx for score, idx in ranked[:num_sentences]])
    
    # Extract the original sentences using the sorted indices
    summary = " ".join([sentences[idx] for idx in top_indices])
    return summary