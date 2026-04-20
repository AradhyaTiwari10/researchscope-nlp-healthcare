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
from src.topic_modeling import get_topics
from typing import List, Dict, Tuple, Any

def process_articles(text_data: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Tuple[int, List[str]]]]:
    """
    Process extracted raw texts mapping summaries back to article URLs, and
    perform topic modeling to extract key terms/themes.
    
    Args:
        text_data (list): List of extracted target dicts:
                          [{"url": "...", "text": "..."}]
                          
    Returns:
        tuple: (Summarized intelligence payload list, List of Topic clusters)
    """
    if not text_data:
        return [], []
        
    valid_data = [
        item for item in text_data 
        if item.get("text") and item.get("url") and len(item["text"].split()) > 60
    ]
    if not valid_data:
        return [], []
        
    print("  [*] Preprocessing texts for NLP...")
    raw_texts = [item["text"] for item in valid_data]
    
    processed_texts = [preprocess(text) for text in raw_texts]
    
    print("  [*] Building global TF-IDF model...")
    try:
        X, global_vectorizer = extract_features(processed_texts)
    except ValueError:
        from sklearn.feature_extraction.text import TfidfVectorizer
        global_vectorizer = TfidfVectorizer()
        X = global_vectorizer.fit_transform(processed_texts)
        
    print("  [*] Performing NLP Topic Modeling...")
    try:
        topics = get_topics(X, global_vectorizer, num_topics=min(3, len(valid_data)), num_words=5)
    except Exception as e:
        print(f"  [-] Topic modeling failed: {e}")
        topics = []

    print("  [*] Generating summaries...")
    results = []
    
    for idx, item in enumerate(valid_data):
        url = item["url"]
        original_text = item["text"]
        
        article_summary = summarize(
            original_text,
            global_vectorizer,
            top_n=3
        )
        
        if len(article_summary.split()) >= 12:
            results.append({
                "url": url,
                "summary": article_summary
            })
        else:
            print(f"  [-] Skipping low-quality/short summary for: {url}")
            
    return results, topics
