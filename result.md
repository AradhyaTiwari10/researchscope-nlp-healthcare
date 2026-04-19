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
"search_results": [],
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
* Updated `run_query.py` to pipe the retrieved dictionary directly into `state["search_results"]`.

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

---

## Phase 3: Content Extraction

### Implementation Details

* Transitioned from raw URLs to full-text document payloads using `newspaper3k`.
* Implemented `extract_article()` to manage per-URL downloading, parsing, and text extraction securely.
* Implemented `extract_multiple()` to loop over `state["search_results"]` specifically, limiting total pulls to 5 valid sources and ignoring failed URLs.
* Updated `run_query.py` to embed the extracted content payload securely into `state["texts"]`.

### Modules Created

* `src/content_extractor.py` → Executes robust URL text-parsing and NLP normalization routines.

### Functions Implemented

* `extract_article(url: str) -> dict` → Attempts article download and parses it into `{"url": url, "text": "..."}`.
* `extract_multiple(search_results: list) -> list` → Filters and extracts multiple results cleanly handling failures gracefully.

### Challenges Handled
- **Failed URLs & Paywalls**: Gracefully skipped via explicit `try-except` blocks without crashing the workflow.
- **Empty Pages**: Text length filters ignore pages that return sterile/empty content.

### Example Output Structure

```json
{
  "texts": [
    {
      "url": "https://example.com/paper",
      "text": "The implementation of ML in early clinical screening..."
    }
  ]
}
```

### Observations & Improvements
- Implemented error handling for inaccessible URLs (e.g., 403 errors)
- Added filtering to skip low-content articles
- Cleaned extracted text to remove excessive whitespace
- Limited extraction to top 3–5 valid articles for efficiency

---

## Phase 4: NLP Processing Integration

### Implementation Details

* Applied the pre-built Classical NLP pipeline to extract meaningful substance dynamically.
* Iterated processed text lists via `nlp_processor` seamlessly wrapping `preprocessing`, `extract_features`, and `summarize` modules together.
* Mapped resultant summaries back to their respective origin `<url>` pointers securely ensuring data pipeline traceability.
* Updated `run_query.py` to complete NLP processing and store payloads dynamically under `state["summaries"]`.

### Modules Created

* `src/nlp_processor.py` → Central routing module that wires raw string extracts into functional math transformers.

### How Global TF-IDF Improves Results
Deploying the shared TF-IDF matrix explicitly (rather than per-document instantiations) lets sequence limits and hyperparameter vocab bounds established upstream be uniformly applied to downstream summarization blocks, assuring consistent sentence rankings logic and robust feature mappings.

### Example Summaries Output

```json
{
  "summaries": [
    {
      "url": "https://example.com/paper",
      "summary": "AI shows massive potential in clinical diagnostics. The accuracy reached 95% in trials. Early detection heavily correlates with survival rates."
    }
  ]
}
```

### Improvements & Optimizations
- Ensured consistency between TF-IDF vocabulary and summarization input
- Fixed vectorizer fallback to properly fit on processed texts
- Added filtering for low-content articles
- Improved summary quality through consistent preprocessing pipeline

---

## Phase 5: LLM Report Generation

### Implementation Details

* Utilized HuggingFace's `transformers` library to initialize `google/flan-t5-base`.
* Dynamically combined isolated classical NLP summaries into a unified token payload seamlessly.
* Drafted a specialized prompt explicitly restricting hallucination by grounding answers purely to the compiled summarized context.
* Restrained inputs using character-limit clipping directly preventing the LLM from overflowing T5 attention boundaries.
* Enforced manual URL sourcing at the end of the output, preventing generic models from hallucinating references completely.
* Updated `run_query.py` to pipe the final outcome locally under `state["report"]`.

### Modules Created

* `src/report_generator.py` → Local LLM text-to-text generative pipeline.

### Prompt Design Approach
Constructed a zero-shot multi-stage instructional prompt demanding specific section outputs (Title, Abstract, Findings, Conclusion). It combines array structures intuitively into string boundaries. 

### Role of LLM vs NLP
* **Classical NLP (Phases 1-4)**: Acted as an exact, deterministic retrieval and extraction mechanism mapping strictly mathematical features.
* **LLM (Phase 5)**: Operates purely as a syntactic synthesizer, shaping the explicit facts generated by the NLP pipeline into readable formats natively.

### Example Output Structure

```text
Title: Advancements in Early Detection
Abstract: Recent studies emphasize machine learning models dynamically scaling diagnostics protocols globally.
Key Findings: 
* ML models detect precise oncology states faster
Conclusion: The integration proves it to be a massive supplementary tool.

Sources:
* https://example.com/ai-health
```

### Improvements & Optimizations
- Implemented lazy loading for HuggingFace model to improve performance
- Strengthened prompt format to enforce structured output
- Reduced dependency on post-processing string replacements
- Added safeguards for empty or weak LLM responses
- Optional deduplication of summaries to improve report clarity
