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
            
            LIMITED_DOMAINS = [
                "pubmed.ncbi.nlm.nih.gov",
                "pmc.ncbi.nlm.nih.gov"
            ]
            
            READABLE_DOMAINS = [
                "medicalnewstoday.com",
                "sciencedaily.com",
                "who.int",
                "mayoclinic.org",
                "cancer.gov",
            ]
            
            def fetch_and_filter(search_string):
                pool = []
                limited_count = 0
                readable_count = 0
                raw_results = list(ddgs.text(search_string, max_results=50))
                if not raw_results:
                    return pool
                
                # First pass: try to get a balanced set
                for res in raw_results:
                    url = res.get("href", "").lower()
                    if url and url not in seen_urls:
                        is_trusted = any(domain in url for domain in TRUSTED_DOMAINS)
                        is_limited = any(domain in url for domain in LIMITED_DOMAINS)
                        is_readable = any(domain in url for domain in READABLE_DOMAINS)
                        
                        if is_trusted:
                            # Rule: Hard cap on overly technical academic archives (PMC/PubMed)
                            if is_limited:
                                if limited_count >= 2:
                                    continue
                                limited_count += 1
                            
                            # Optimization: If we have 4 results and 0 are readable,
                            # skip this one if it's NOT readable (unless we're at the very end of raw_results)
                            if len(pool) == 4 and readable_count == 0 and not is_readable:
                                continue

                            if is_readable:
                                readable_count += 1

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
