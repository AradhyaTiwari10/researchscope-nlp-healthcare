"""
run_agent.py
------------
CLI Entry Point for Phase 6 Workflow
Replaces procedural run_query.py sequence with agent graph execution.
"""

from src.agent_graph import run_agent

def main():
    print("====================================")
    print(" ResearchScope NLP Healthcare Agent ")
    print("====================================")
    
    query = input("\nEnter your research query:\n> ").strip().lstrip(">").strip()
    
    if not query:
        print("No query entered. Exiting.")
        return
        
    print(f"\n🚀 Initiating LangGraph Workflow for: '{query}'\n")
    
    try:
        final_state = run_agent(query)
        
        print("\n\n" + "="*50)
        print("FINAL RESEARCH REPORT")
        print("="*50 + "\n")
        print(final_state.get("report", "Report generation failed. Empty payload returned."))
        print("\n" + "="*50)
        
    except Exception as e:
        print(f"\n❌ Workflow Failed: {e}")

if __name__ == "__main__":
    main()
