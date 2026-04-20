"""
content_extractor.py
--------------------
Module: Content Extraction
Phase 3 – ResearchScope NLP Healthcare

Responsibilities:
  - Download and parse web article content via URLs.
  - Handle extraction failures gracefully.
  - Return clean text aligned for downstream NLP workflows.

Public API:
  extract_article(url: str) -> dict
  extract_multiple(search_results: list) -> list
"""

from newspaper import Article, Config

def extract_article(url: str) -> dict:
    """
    Download and extract the main text from a given article URL.
    
    Args:
        url (str): The URL of the web article.
        
    Returns:
        dict: Containing the 'url' and the extracted 'text'.
    """
    try:
        print(f"  [+] Downloading: {url}")
        
        custom_config = Config()
        custom_config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        custom_config.request_timeout = 15
        
        article = Article(url, config=custom_config)
        article.download()
        article.parse()
        text = article.text.strip()
        
        if not text:
            print(f"  [-] Warning: Extracted text is empty for {url}")
            return { "url": url, "text": "" }
            
        text = " ".join(text.split())
        
        if len(text.split()) < 60:
            print(f"  [-] Warning: Extracted text too short (<60 words) for {url}")
            return { "url": url, "text": "" }
            
        return {
            "url": url,
            "text": text
        }
    except Exception as e:
        print(f"  [x] Error extracting {url}: {e}")
        return {
            "url": url,
            "text": ""
        }

def extract_multiple(search_results: list, max_articles: int = 8) -> list:
    """
    Iterate over search results and extract articles, skipping failures 
    and capping the total downloaded.
    
    Args:
        search_results (list): List of dictionaries from search_web().
        max_articles (int): Maximum number of valid articles to extract.
        
    Returns:
        list: A list of dictionaries containing URLs and their text.
    """
    extracted_articles = []
    
    for res in search_results:
        url = res.get("url")
        if not url:
            continue
            
        result = extract_article(url)
        
        if result["text"]:
            extracted_articles.append(result)
            
        if len(extracted_articles) >= max_articles:
            break
            
    return extracted_articles[:8]
