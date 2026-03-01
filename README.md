# ResearchScope: Traditional NLP Research Analysis System
## Topic: Artificial Intelligence and Machine Learning in Healthcare
### University Capstone — Milestone 1

---

## ⚙️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│                   PDF Research Papers                │
│                   (data/raw/*.pdf)                   │
└───────────────────────┬─────────────────────────────┘
                        │  pdfplumber (page-by-page)
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Text Extraction                     │
│          Empty page filtering, corpus join           │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│               Preprocessing Pipeline                 │
│   Lowercase → Rm Numbers → Rm Punct → spaCy Tokens  │
│   → Stopword Filter (NLTK + Domain) → Lemmatize     │
│   → Length Filter → Bigram Detection (Phrases)      │
└───────────────────────┬─────────────────────────────┘
                        │
               ┌────────┴────────┐
               ▼                 ▼
┌──────────────────────┐ ┌───────────────────────────┐
│   TF-IDF Vectors     │ │   Gensim BoW Corpus        │
│  (scikit-learn)      │ │  (Dictionary + doc2bow)    │
└──────────┬───────────┘ └──────────┬────────────────┘
           │                        │
           ▼                        ▼
┌──────────────────────┐ ┌───────────────────────────┐
│  Keyword Extraction  │ │   LDA Topic Modeling       │
│  (Document & Global) │ │   (K=2–10 topics)          │
└──────────────────────┘ └──────────┬────────────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │  Coherence Evaluation │
                         │  (C_V metric)         │
                         └──────────────────────┘

┌─────────────────────────────────────────────────────┐
│            Extractive Summarization                  │
│   Noise Cleaning → Sent Tokenize → TF-IDF Rank      │
│   → Top 5 Sentences (original order)                │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│  Metrics | Keywords | Topics | Coherence | Summary   │
└─────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
researchscope-nlp-healthcare/
│
├── data/
│   ├── raw/         # Project 5: Customer Churn Prediction & Agentic Retention Strategy (Milestone 1)

## From Predictive Analytics to Intelligent Intervention

### Project Overview
This project involves the design and implementation of an **AI-driven customer analytics system** that predicts customer churn and evolves into an agentic AI retention strategist. Currently, the project is structured at **Milestone 1**, focusing strictly on classical NLP and Machine Learning techniques.

- **Milestone 1 (Current):** Classical NLP pipeline (TF-IDF, LDA) applied to a dataset of Healthcare AI research papers to extract foundational topics and generate extractive summaries.
- **Milestone 2 (Future):** Extension into an agent-based AI application that autonomously reasons about risk, retrieves retention best practices (RAG), and plans intervention strategies.

---

### Constraints & Requirements
- **Team Size:** 3–4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** Streamlit (UI), Scikit-Learn (ML), NLTK/spaCy (NLP)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **Data Ingestion** | PyPDF2 |
| **NLP Preprocessing** | NLTK, spaCy, RegEx |
| **ML Models (M1)** | TF-IDF Vectorizer, Latent Dirichlet Allocation (scikit-learn) |
| **UI Framework** | Streamlit |
| **Deployment** | Upcoming (M2) |

---

### Milestones & Deliverables

#### Milestone 1: Classical NLP Topic Modeling (Mid-Sem - Achieved)
**Objective:** Identify key topics and summarize healthcare AI research using purely classical NLP pipelines *without LLMs*.

**Key Deliverables:**
- Problem understanding & Business context.
- System architecture diagram (in code comments & logic flow).
- Working local application with UI (Streamlit).
- **Extractive Summarization:** 3-sentence scoring based on TF-IDF.
- **Topic Modeling:** 4 extracted latent topics representing document corpura.

#### Milestone 2: Agentic AI Assistant (End-Sem - Upcoming)
**Objective:** Extend the system into an agentic strategist that reasons via LangGraph/RAG.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured retention report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering, UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Reasoning quality, RAG & State management implementation, Output clarity, Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. Milestone 2 project must be hosted.
tone 1 prototype** using traditional NLP only. The following known limitations apply:

| Limitation | Technical Explanation |
|:---|:---|
| **Extractive summaries lack abstraction** | Sentences are pulled verbatim from the document. The system cannot paraphrase or synthesize cross-document knowledge. |
| **LDA struggles with small corpora** | With < 15 documents, the Dirichlet prior has insufficient mass to reliably separate topics. Results improve with larger corpora. |
| **TF-IDF ignores semantic similarity** | "Cardiovascular" and "cardiac" are treated as completely unrelated tokens. No vector space proximity is used. |
| **No contextual embeddings** | Representations are bag-of-words. Word order, polysemy, and context-dependent meaning are not captured. |
| **PDF extraction noise** | Multi-column layouts, footnote bleeding, and figure captions in academic PDFs introduce noisy tokens that require extensive rule-based cleaning. |

---

## Optimization Strategy

The system employs several classical strategies for quality improvement:

1. **Hyperparameter tuning for K**: Coherence scores are computed over K = 2–10, allowing data-driven selection of the optimal topic count.
2. **Stopword refinement**: A merged NLTK + domain-specific stoplist removes both common English stop words and domain artifacts (figure labels, DOIs, PDF metadata tokens).
3. **Vocabulary filtering**: `dictionary.filter_extremes(no_below=2, no_above=0.7)` removes singleton and over-common terms before LDA training.
4. **Bigram detection**: `gensim.models.Phrases` surfaces compound terms (e.g., `machine_learning`, `neural_network`) that would otherwise be split and dilute topic coherence.
5. **Coherence-driven model selection**: The system selects K that maximizes the C_V coherence metric — a standard proxy for human topic interpretability.

---

## Constraints

- No OpenAI, Gemini, Claude, or any LLM
- No Transformers or deep learning models
- Traditional NLP only: `nltk`, `spacy`, `scikit-learn`, `gensim`