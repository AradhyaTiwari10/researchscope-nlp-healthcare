"""
agent_state.py
---------------
Module: Agent State Management
Phase 1 – ResearchScope NLP Healthcare

Responsibilities:
  - Acquire research query from user
  - Initialize the shared state object for the agent workflow

Public API:
  get_user_query() -> str
  initialize_state(query: str) -> dict
"""

def get_user_query() -> str:
    """
    Prompt the user for a research query via the console.
    
    Returns:
        str: The user's query string.
    """
    try:
        query = input('Input:\n')
        return query.strip()
    except EOFError:
        return ""

def initialize_state(query: str) -> dict:
    """
    Initialize the starting state for the agentic workflow.
    
    Args:
        query (str): The primary research question or topic.
        
    Returns:
        dict: The state dictionary object that will flow through the agent nodes.
    """
    state = {
        "query": query,
        "search_results": [],
        "texts": [],
        "summaries": [],
        "report": ""
    }
    return state
