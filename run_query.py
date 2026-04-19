"""
run_query.py
-------------
Entry script for testing Agent query input and state initialization.
Phase 1 – ResearchScope NLP Healthcare
"""

import json
from src.agent_state import get_user_query, initialize_state

def main():
    # 1. Get user query
    query = get_user_query()
    
    if not query:
        print("No query entered. Exiting.")
        return

    # 2. Initialize agent workflow state
    state = initialize_state(query)
    
    # 3. Output expectation match
    print("\nOutput:")
    print(json.dumps(state, indent=2))

if __name__ == "__main__":
    main()
