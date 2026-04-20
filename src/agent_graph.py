"""
agent_graph.py
--------------
Module: LangGraph Agent Workflow
Phase 6 – ResearchScope NLP Healthcare

Responsibilities:
  - Wrap standalone functions into LangGraph nodes.
  - Define explicit graph edges traversing the intelligence pipeline.
  - Compile the state execution workflow safely.

Public API:
  run_agent(query: str) -> dict
"""

from langgraph.graph import StateGraph, START, END
from src.web_search import search_web
from src.content_extractor import extract_multiple
from src.nlp_processor import process_articles
from src.report_generator import generate_report
from src.utils import is_medical_query



def check_scope_node(state: dict) -> dict:
    """Classifies query intent before search."""
    print("  => [Node: ScopeGuard] Checking medical intent...")
    state["is_medical"] = is_medical_query(state["query"])
    return state

def rejection_node(state: dict) -> dict:
    """Returns a rejection message for out-of-scope queries."""
    print("  => [Node: Rejection] Query classified as non-medical.")
    state["report"] = (
        "OUT OF SCOPE: The ResearchScope agent is specialized for medical and healthcare research.\n"
        "Please enter a query related to diseases, treatments, clinical trials, or health breakthroughs."
    )
    return state

def search_node(state: dict) -> dict:
    """Invokes DuckDuckGo dynamically based on the state query."""
    print("  => [Node: Search] Activating retrieval tools...")
    state["search_results"] = search_web(state["query"])
    return state

def extract_node(state: dict) -> dict:
    """Downloads raw HTML constraints via Newspaper3k."""
    print("  => [Node: Extract] Pulling article structures...")
    state["texts"] = extract_multiple(state["search_results"])
    return state

def process_node(state: dict) -> dict:
    """Analyzes text vectors mathematically compiling sentences to summaries."""
    print("  => [Node: Process] Generating NLP vectors & summaries...")
    summaries, topics = process_articles(state["texts"])
    state["summaries"] = summaries
    state["topics"] = topics
    return state

def report_node(state: dict) -> dict:
    """Triggers LLM formatting constraint execution."""
    print("  => [Node: Report] Synthesizing final structured text...")
    state["report"] = generate_report(state["query"], state["summaries"])
    return state


builder = StateGraph(dict)

builder.add_node("check_scope", check_scope_node)
builder.add_node("rejection", rejection_node)
builder.add_node("search", search_node)
builder.add_node("extract", extract_node)
builder.add_node("process", process_node)
builder.add_node("report", report_node)

def route_intent(state: dict):
    if state.get("is_medical"):
        return "medical"
    return "out_of_scope"

builder.set_entry_point("check_scope")

builder.add_conditional_edges(
    "check_scope",
    route_intent,
    {
        "medical": "search",
        "out_of_scope": "rejection"
    }
)

builder.add_edge("search", "extract")
builder.add_edge("extract", "process")
builder.add_edge("process", "report")

builder.add_edge("report", END)
builder.add_edge("rejection", END)

graph = builder.compile()


def run_agent(query: str) -> dict:
    """
    Initializes standard memory state and invokes the compiled LangGraph workflow seamlessly.
    """
    state = {
        "query": query,
        "search_results": [],
        "texts": [],
        "summaries": [],
        "topics": [],
        "report": "",
        "is_medical": False
    }
    
    result = graph.invoke(state)
    return result
