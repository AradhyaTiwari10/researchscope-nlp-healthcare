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
- Optional deduplication of summaries to improve report clarity
- Implemented robust search retry logic with query simplification to handle strict indexing or rate limiting
- Resolved HuggingFace pipeline `text2text-generation` task errors by utilizing architectural auto-inference
- Fixed SSL and 403 Forbidden errors in extraction by spoofing modern browser User-Agents and enforcing US-English search regions where applicable

---

## Phase 6: LangGraph Agent Workflow

### Implementation Details

* Transitioned entirely from basic procedural Python scripts (`A -> B -> C`) into a formal agentic state machine using `langgraph`.
* Migrated the unified pipeline configuration (Search -> Extract -> NLP -> Report) firmly into explicit `StateGraph` isolated nodes.
* Defined sequential edges binding node relationships dynamically.
* Abstracted the complex multi-module orchestration strictly under a compiled `run_agent(query)` invocation loop.
* Replaced linear execution (`run_query.py`) completely with a dedicated agent-based workflow payload (`run_agent.py`).

### Modules Created

* `src/agent_graph.py` → Constructs exact nodes, binds edges, enforces exact state formatting, and invokes compiled graphs dynamically.
* `run_agent.py` → Replaces `run_query.py` entirely, triggering graph traversals inherently based purely on query mapping.

### Why LangGraph is Better Than a Linear Pipeline
While procedural calls handle straightforward sequences cleanly, a dedicated State Graph manages the `state` dictionary entirely as **Memory** rather than merely swapping static variables recursively. It inherently provides a stable framework for future scaling:
* **Conditional Routing:** LangGraph enables cycles directly—for example, looping back to the 'search' node safely if the 'extract' node fails to parse any data.
* **Checkpointing & Fault Tolerance:** LangGraph's architecture enables deep checkpointing capabilities inherently, allowing processes to halt, ask for explicit manual human intervention via `interrupt`, and organically resume previously instantiated state nodes smoothly without redundant generation.
* **Granular Observability:** Isolating logic natively inside specific 'Nodes' inherently exposes state execution telemetry mapping securely.

### Phase 8 - Scope Enforcement & Testing
- Integrated **Medical Scope Guard** using LangGraph conditional routing
- Implemented a validator to reject non-healthcare/non-medical queries
- Established a **Test Suite** to verify intent classification and trusted sourcing

---

## 🧪 Testing Guide

To verify the system, run `python3 run_agent.py` and try these test cases:

### 1. Verification of Medical Scope (Accepted)
Try these queries to see how the agent pulls high-quality medical data:
- `Clinical trials for lung cancer immunotherapy`
- `Recent breakthroughs in cardiology and heart failure 2024`
- `WHO reports on infectious disease prevention`

### 2. Verification of Scope Guard (Rejected)
Try these queries to see how the system protects its medical specialization:
- `Who won the FIFA World Cup 2022?`
- `Latest geopolitical tensions between USA and Iran`
- `How to bake a chocolate cake`

### 3. Verification of Trusted Metrics
Observe the logs to see `[Node: Search]` filtering for domains like `.nih.gov`, `.who.int`, and `.cancer.gov`. Non-trusted sources are automatically ignored.

---

### 🚀 Final Output Quality Enhancements
- **Hybrid Summarization**: Implemented TF-IDF scoring on processed text while extracting ORIGINAL sentences for maximum readability.
- **Balanced Sourcing**: Restricted overly technical papers (PubMed/PMC) to 1-2 per report to prioritize readable clinical news.
- **Strict Prompt Engineering**: Enforced simplified language and structured markdown for readable, professional reports.
- **Quality Filters**: Discarded low-quality summary fragments (<20 words) to prevent "academic noise" in final reports.
- **Human-Centric Formatting**: Refined LLM behavior to avoid technical jargon and focus on real-world treatment impacts.

---

### 🚀 Final Depth Enhancements
- **Maximizing Coverage**: Increased the article intake limit from 5 to 8 to provide a broader context pool for research synthesis.
- **Smart Sensitivity**: Relaxed word-count thresholds (from 100 to 60 for raw text; 20 to 12 for summaries) to prevent "data starvation" and preserve valuable mid-sized clinical insights.
- **Dense Context Aggregation**: Implemented summary merging to feed the LLM a unified, information-dense block instead of isolated fragments.
- **Forced Narrative Expansion**: Strengthened prompt engineering to explicitly command longer, multi-sentence sections and more detailed bullet point findings.

---

### 🚀 Final LLM Optimization — Multi-Step Prompt Decomposition
- **Root cause identified**: Flan-T5-base has a ~512 token context window and compresses everything aggressively when given a single large structured prompt.
- **Solution**: Replaced single-pass generation with **prompt decomposition** — each report section (Abstract, Key Findings, Conclusion) is now generated independently by its own focused prompt, then assembled manually.
- **Why it works**: Flan-T5 excels at small, focused tasks. Decomposing avoids context overflow and enables richer per-section output.
- **Output improvement**: Report output depth increased from 1-2 sentences to structured multi-paragraph reports.

---

### 🚀 Migration: Flan-T5 → Groq (llama3-70b-8192)
- **Replaced** local HuggingFace Flan-T5-base with Groq-hosted `llama3-70b-8192` model.
- **Why**: Flan-T5 has a ~512 token window and compresses outputs aggressively. `llama3-70b` has an 8192-token window and produces vastly superior structured reports.
- **Architecture preserved**: Multi-step prompt decomposition maintained — Abstract, Key Findings, and Conclusion still generated independently for guaranteed structure.
- **Cost**: Free-tier Groq API (no credit card). Suitable for demos and capstone evaluations.
- **Speed**: Groq inference is ~10x faster than local CPU inference.

---

### 💎 Final Output Refinements
- **Sanitized Query Input**: Stripped formatting artifacts like `>` and redundant whitespace from terminal inputs to ensure clean report titles.
- **Noise Elimination**: Implemented post-generation cleaning to remove redundant LLM-generated headers (e.g., "Here are 6 key findings...").
- **Proactive Simplification**: Added explicit instructions to all LLM prompts to summarize discoveries in plain language and avoid unnecessary technical verbosity.
- **Source Balancing**: Integrated a diversity enforcer in the search node to guarantee at least one patient-centric source (e.g., Mayo Clinic, Medical News Today) alongside academic papers.

---

## Phase 7: Streamlit UI
- Built interactive UI for research assistant
- Integrated LangGraph agent into frontend
- Added structured report rendering
- Implemented loading states and error handling
- Prepared system for deployment

## End-Sem Deliverables (Final Milestone 2)
### ✅ Agentic AI Workflow using LangGraph 
- Implemented robust multi-step control flow (ScopeGuard → Search → Extract → Process → Report).
- Explicit state management implemented across workflow nodes.
- Handle APIs correctly (skips 403 blocks dynamically, handles Groq AI limits natively).

### ✅ Structured Output & NLP Features 
- Extracted and integrated **Topics Clusters** and **Key Terms** seamlessly using LDA algorithm via backend tf-idf scores. 
- Original **Extractive Summaries** directly mapped to source URLs for transparency.
- Full structural output returned correctly (Title, Abstract, Key Findings, Conclusion, Sources).

### ✅ Enhanced UI & Extension
- Streamlit UI acts as the unified frontend bridging user query to agent execution.
- Added visual staggered status indicators for loading pipelines.
- **Extension Chosen:** PDF Export. Developed on-the-fly markdown-to-PDF rendering capabilities integrating  directly inside the frontend component to let operators easily download health reports securely offline.


## End-Sem Deliverables (Final Milestone 2)
### ✅ Agentic AI Workflow using LangGraph
- Implemented robust multi-step control flow (ScopeGuard -> Search -> Extract -> Process -> Report).
- Explicit state management implemented across workflow nodes.
- Handle APIs correctly (skips 403 blocks dynamically, handles Groq AI limits natively).

### ✅ Structured Output & NLP Features
- Extracted and integrated **Topics Clusters** and **Key Terms** seamlessly using LDA algorithm via backend tf-idf scores.
- Original **Extractive Summaries** directly mapped to source URLs for transparency.
- Full structural output returned correctly (Title, Abstract, Key Findings, Conclusion, Sources).

### ✅ Enhanced UI & Extension
- Streamlit UI acts as the unified frontend bridging user query to agent execution.
- Added visual staggered status indicators for loading pipelines.
- **Extension Chosen:** PDF Export. Developed on-the-fly markdown-to-PDF rendering capabilities integrating `fpdf` directly inside the frontend component to let operators easily download health reports securely offline.

## Continuous Integration (CI)
### ✅ GitHub Actions setup
- Developed a basic CI/CD pipeline in `.github/workflows/ci.yml`.
- Workflow is triggered on push and pull requests to the `main` branch.
- Added `flake8` to `requirements.txt` to integrate Python code quality checks into the pipeline.
