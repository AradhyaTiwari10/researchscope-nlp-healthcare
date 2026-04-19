# ResearchScope NLP Healthcare - Project Tracking

## Phase 0: NLP Pipeline Refactoring

**Implementation Details:**
- Refactored a monolithic Streamlit application into a modular, production-ready backend architecture.
- Abstracted isolated modules with clear, single-responsibility functions for seamless future agent integration.
- Extracted and enhanced `preprocess`, `extract_features`, `get_topics`, `summarize`, and `compute_similarity` functionalities to operate independently of the UI.
- Implemented `test_pipeline.py` to ensure module integrability and end-to-end functionality.

**Modules Created/Updated:**
- `src/preprocessing.py`: Houses advanced regex, ligature fixing, tokenization, and lemmatization logic.
- `src/feature_engineering.py`: Computes TF-IDF features dynamically based on modular input.
- `src/topic_modeling.py`: Connects NLP features with Latent Dirichlet Allocation (LDA) for thematic clustering.
- `src/summarizer.py`: Performs extractive, content-rich summarization leveraging TF-IDF sentence ranking.
- `src/similarity.py`: Implements cosine similarity to find relations between disparate pieces of research data.

**Functions Added:**
- `preprocessing.preprocess(text)`
- `feature_engineering.extract_features(texts)`
- `topic_modeling.get_topics(X, vectorizer)`
- `summarizer.summarize(text, vectorizer)`
- `similarity.compute_similarity(X)`

**Architecture Explanation:**
The system previously coupled core NLP business logic tightly with Streamlit UI directives. This refactoring phase implements a Service-Oriented structure. Each phase of the Classical NLP pipeline is now wrapped into a standalone, stateless API function nested under the `src/` domain. This setup allows AI agents, API wrappers (like FastAPI), or external task schedulers to import directly from the source logic without spinning up UI dependencies or executing monolithic global states, greatly enhancing portability, testability, and scalability.

---

## Phase 1: Query-Based Input

### Implementation Details

* Transitioned system input from static document ingestion to dynamic, query-driven interaction.
* Implemented CLI-based query handler to simulate real user input.
* Introduced a structured state object to act as the central data carrier across the agent pipeline.
* Created `run_query.py` as the execution entry point for initializing agent workflows.

---

### Modules Created

* `src/agent_state.py` → Handles query input and state initialization

---

### Functions Implemented

* `get_user_query()` → Captures user query from CLI
* `initialize_state(query)` → Initializes agent state dictionary

---

### State Object Design

The system introduces a unified state object:

```json
{
"query": "...",
"urls": [],
"texts": [],
"summaries": [],
"report": ""
}
```

This state acts as the **single source of truth** throughout the pipeline.

---

### Role in Agent Workflow

* Serves as the input payload for all future processing nodes
* Enables sequential state updates across:

  * Web retrieval
  * Text extraction
  * NLP processing
  * Report generation
* Eliminates dependency on static inputs like PDFs

---

### Execution Flow (Current)

User Input → get_user_query() → initialize_state() → print(state)

---

### Testing & Validation

* Successfully executed `run_query.py`
* Verified correct state initialization
* Output matches expected structure with empty placeholders

---

### Outcome

The system is now:

* Query-driven ✅
* State-aware ✅
* Ready for web integration ✅
* Compatible with LangGraph workflows ✅

---

## Phase 2: Web Search Integration

### Implementation Details

* Integrated DuckDuckGo Search API to handle web retrieval without requiring authentication.
* Created a reusable, stateless searching module designed specifically to fetch context for research queries.
* Handled URL deduplication, error catching, and limiting output to the top 5 results to keep context concise for the LLM.
* Updated `run_query.py` to pipe the retrieved dictionary directly into `state["urls"]`.

### Modules Created

* `src/web_search.py` → Executes secure, network-safe searches via DDGS.

### Functions Implemented

* `search_web(query: str) -> list` → Returns a structured list containing dictionaries with `title`, `url`, and `snippet`.

### Example Output Structure

```json
[
  {
    "title": "A Review of AI in Cancer Detection",
    "url": "https://example.com/paper",
    "snippet": "This study explores how early..."
  }
]
```

### Role in Agent Workflow

This phase officially transitions the pipeline from a sterile input state to a Retrieval-Augmented Generation (RAG) agent. By fetching top real-world results dynamically based on user prompts, the workflow acts as an AI Researcher that gathers its own context explicitly before doing extractive analysis or summarization.
