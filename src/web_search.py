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
    """
    results = []
    seen_urls = set()
    
    TRUSTED_DOMAINS = [
        "nih.gov", "who.int", "mayoclinic.org", "cancer.gov",
        "medicalnewstoday.com", "sciencedaily.com", "nature.com",
        "thelancet.com", "bmj.com", "webmd.com", "healthline.com"
    ]
    
    try:
        with DDGS() as ddgs:
            def fetch_and_filter(search_string, strict=True):
                pool = []
                try:
                    # ddgs 8.x+ best practice: specify region and max_results
                    raw_results = list(ddgs.text(search_string, region='wt-wt', max_results=25))
                except Exception as e:
                    print(f"  [!] DDGS Search Error: {e}")
                    return []

                if not raw_results:
                    return pool
                
                for res in raw_results:
                    url = res.get("href", "").lower()
                    if url and url not in seen_urls:
                        is_trusted = any(domain in url for domain in TRUSTED_DOMAINS)
                        
                        if is_trusted or not strict:
                            seen_urls.add(url)
                            pool.append({
                                "title": res.get("title", ""),
                                "url": url,
                                "snippet": res.get("body", "")
                            })
                            if len(pool) >= 5:
                                break
                return pool
            
            # 1. Try Strict Search (Academic Sources)
            results = fetch_and_filter(query, strict=True)
            
            # 2. Fallback to General Search
            if not results:
                results = fetch_and_filter(query, strict=False)
                
            # 3. Fallback to Simplified Keywords
            if not results:
                simple = " ".join(query.split()[:3])
                results = fetch_and_filter(simple, strict=False)
                
    except Exception as e:
        print(f"⚠️ Search Module Error: {e}")
        
    return results
