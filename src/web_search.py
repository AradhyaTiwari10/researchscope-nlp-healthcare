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

from duckduckgo_search import DDGS
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
            # We fetch up to 10 to ensure we get 5 valid, unique results
            # We fetch up to 10 to ensure we get 5 valid, unique results
            raw_results = ddgs.text(query, max_results=10)
            
            if not raw_results:
                print(f"\n  [!] DuckDuckGo returned 0 results for this query.")
                print(f"      (Try using core keywords instead of conversational sentences like 'latest cancer research')\n")
                return []
                
            for res in raw_results:
                url = res.get("href", "")
                
                # Filter out invalid or duplicate URLs
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "title": res.get("title", ""),
                        "url": url,
                        "snippet": res.get("body", "")
                    })
                    
                if len(results) >= 5:
                    break
                    
    except Exception as e:
        print(f"⚠️ Error during web search: {e}")
        
    return results
