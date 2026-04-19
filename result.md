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

**Implementation Details:**
- Added query-driven input prompt to accept dynamic user requests via CLI.
- Introduced an extensible state dictionary designed for the broader LangGraph/agentic framework.
- Created `run_query.py` script to simulate the entry point for the agent logic.

**Functions Added:**
- `agent_state.get_user_query()`
- `agent_state.initialize_state(query)`

**Purpose of State Object:**
The `state` object encapsulates essential parameters:
`{"query": "...", "urls": [], "texts": [], "summaries": [], "report": ""}`
It acts as the central shared memory and message-passing payload across the upcoming node-based agent workflow. Each processing node in Phase 2+ can read from it concurrently, append context (urls, text extracts), and ultimately compile a final report.

**How This Enables Agent Workflow:**
By modularizing the state and accepting queries upfront, we eliminate reliance on static, local document processing constraints. The system is now primed for web-retrieval and multi-step pipeline chaining characteristic of LangChain/LangGraph architectures, because initial triggers and data sources are cleanly decoupled.
