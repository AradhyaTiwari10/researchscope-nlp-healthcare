"""
run_query.py
-------------
Entry script for testing Agent query input and state initialization.
Phase 1 – ResearchScope NLP Healthcare
"""

import json
from src.agent_state import get_user_query, initialize_state
from src.web_search import search_web
from src.content_extractor import extract_multiple

def main():
    # 1. Get user query
    query = get_user_query()
    
    if not query:
        print("No query entered. Exiting.")
        return

    # 2. Initialize agent workflow state
    state = initialize_state(query)
    
    print("\n🔍 Executing Web Search...")
    # 3. Call search_web and store in state
    search_results = search_web(state["query"])
    state["search_results"] = search_results
    
    # 4. Extract content
    print("\n📄 Extracting Content...")
    extracted_data = extract_multiple(state["search_results"])
    state["texts"] = extracted_data
    
    print(f"\n=> Successfully extracted {len(extracted_data)} articles.")
    
    # 5. Output expectation match
    print("\nOutput:")
    # We clip the actual text payload for standard output console scrolling so it doesn't flood the terminal
    state_preview = state.copy()
    if state_preview["texts"]:
        for text_doc in state_preview["texts"]:
            if len(text_doc["text"]) > 100:
                text_doc["text"] = text_doc["text"][:100] + "... [TRUNCATED FOR PRINT]"
                
    print(json.dumps(state_preview, indent=2))

if __name__ == "__main__":
    main()
