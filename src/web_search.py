"""
web_search.py
--------------
Module: Web Search Integration
Phase 2 – ResearchScope NLP Healthcare

Responsibilities:
  - Retrieve relevant web results using a research query.
  - Utilize DuckDuckGo (DDGS) securely without API keys.
  - Deduping, error-handling, and extracting specific fields.

Public API:
  search_web(query: str) -> list
"""

from ddgs import DDGS
from typing import List, Dict

def search_web(query: str) -> List[Dict[str, str]]:
    """
    Search the web for a given query and return the top 5 unique results.
    
    Args:
        query (str): The search phrase to execute.
        
    Returns:
        List[Dict[str, str]]: A list of up to 5 dictionaries containing:
                              'title', 'url', and 'snippet'.
    """
    results = []
    seen_urls = set()
    
    try:
        with DDGS() as ddgs:
            TRUSTED_DOMAINS = [
                "nih.gov",
                "who.int",
                "mayoclinic.org",
                "cancer.gov",
                "medicalnewstoday.com",
                "sciencedaily.com",
                "nature.com",
                "thelancet.com"
            ]
            
            def fetch_and_filter(search_string):
                pool = []
                raw_results = list(ddgs.text(search_string, max_results=50))
                if not raw_results:
                    return pool
                for res in raw_results:
                    url = res.get("href", "")
                    if url and url not in seen_urls:
                        is_trusted = any(domain in url.lower() for domain in TRUSTED_DOMAINS)
                        if is_trusted:
                            seen_urls.add(url)
                            pool.append({
                                "title": res.get("title", ""),
                                "url": url,
                                "snippet": res.get("body", "")
                            })
                            if len(pool) >= 5:
                                break
                return pool
                
            results = fetch_and_filter(query)
            
            # If the strict trusted domain filter stripped all results or query returned 0, try simplified keywords
            if not results:
                simplified_query = " ".join(query.split()[:4])
                if simplified_query != query:
                    print(f"  [!] Trusted results empty for full query. Retrying with keywords: '{simplified_query}'...")
                    results = fetch_and_filter(simplified_query)
            
            if not results:
                print(f"\n  [!] DuckDuckGo returned 0 TRUSTED results for this query.")
                print(f"      (Current strict filters active. Try adjusting keywords)\n")
                return []
                    
    except Exception as e:
        print(f"⚠️ Error during web search: {e}")
        
    return results
