"""
nlp_processor.py
----------------
Module: NLP Processing Integration
Phase 4 – ResearchScope NLP Healthcare

Responsibilities:
  - Generate core NLP features (TF-IDF) from raw web texts.
  - Summarize extracted web articles.
  - Map specific summaries securely back to corresponding URLs.

Public API:
  process_articles(text_data: list) -> list
"""

from src.preprocessing import preprocess
from src.feature_engineering import extract_features
from src.summarizer import summarize
from typing import List, Dict

def process_articles(text_data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Process extracted raw texts mapping summaries back to article URLs.
    
    Args:
        text_data (list): List of extracted target dicts:
                          [{"url": "...", "text": "..."}]
                          
    Returns:
        list: Summarized intelligence payload:
              [{"url": "...", "summary": "..."}]
    """
    if not text_data:
        return []
        
    # 1. Filter out empty or misformatted entries
    valid_data = [
        item for item in text_data 
        if item.get("text") and item.get("url") and len(item["text"].split()) > 100
    ]
    if not valid_data:
        return []
        
    print("  [*] Preprocessing texts for NLP...")
    raw_texts = [item["text"] for item in valid_data]
    
    # 2. Preprocess string payloads
    processed_texts = [preprocess(text) for text in raw_texts]
    
    # 3. Generate Global TF-IDF
    print("  [*] Building global TF-IDF model...")
    try:
        # Utilizing the common pipeline parameters
        X, global_vectorizer = extract_features(processed_texts)
    except ValueError:
        # Revert to standard initialization if empty vocabulary arises
        from sklearn.feature_extraction.text import TfidfVectorizer
        global_vectorizer = TfidfVectorizer()
        global_vectorizer.fit(processed_texts)

    # 4. Generate Summaries
    print("  [*] Generating summaries...")
    results = []
    
    for idx, item in enumerate(valid_data):
        url = item["url"]
        original_text = item["text"]
        
        # Summarize logic uses TF-IDF to score internal processed mapping but extracts original sentences natively
        article_summary = summarize(
            original_text,
            global_vectorizer,
            top_n=3
        )
        
        results.append({
            "url": url,
            "summary": article_summary
        })
        
    return results
