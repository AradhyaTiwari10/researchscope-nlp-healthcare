# Project: NLP Research Analysis System & Agentic Healthcare Strategist

## From Classical Text Mining to Intelligent Assistants

### Project Overview
This project involves the design and implementation of an **AI-driven NLP analytics system** that ingests, summarizes, and discovers latent topics across Healthcare AI research papers, and evolves into an intelligent research assistant.

- **Milestone 1:** Classical Natural Language Processing (NLP) and Machine Learning techniques (TF-IDF, Latent Dirichlet Allocation) applied to PDF research documents to extract clean summaries and identify key thematic drivers.
- **Milestone 2:** Extension into an agent-based AI application that autonomously reasons about the research topics, retrieves information using RAG (Retrieval-Augmented Generation), and interacts comprehensively with the document corpus.

---

### Constraints & Requirements
- **Team Size:** 3–4 Students
- **API Budget:** Free Tier Only (Open-source models / Free APIs)
- **Framework:** LangGraph (Recommended for M2), Streamlit (UI)
- **Hosting:** Mandatory (Hugging Face Spaces, Streamlit Cloud, or Render)

---

### Technology Stack
| Component | Technology |
| :--- | :--- |
| **Data Ingestion** | PyPDF2 |
| **ML Models & NLP (M1)** | Scikit-Learn (TF-IDF, LDA), NLTK, spaCy |
| **Agent Framework (M2)** | LangGraph, Chroma/FAISS (RAG) |
| **UI Framework** | Streamlit |
| **LLMs (M2)** | Open-source models or Free-tier APIs |

---

### Milestones & Deliverables

#### Milestone 1: Classical NLP Topic Modeling (Mid-Sem)
**Objective:** Identify underlying research topics and generate extractive summaries using classical NLP pipelines *without LLMs*.

**Key Deliverables:**
- Problem understanding & Business context.
- System architecture diagram (Code structure logic).
- Working local application with UI (Streamlit).
- Model performance evaluation report (Cohesive and distinct topics, 3-sentence summary extraction rules).

#### Milestone 2: Agentic AI Research Assistant (End-Sem)
**Objective:** Extend the system into an agentic strategist that reasons about the research topics, retrieves best practices/document contexts (RAG), and generates structured insights.

**Key Deliverables:**
- **Publicly deployed application** (Link required).
- Agent workflow documentation (States & Nodes).
- Structured research insight report generation.
- GitHub Repository & Complete Codebase.
- Demo Video (Max 5 mins).

---

### Evaluation Criteria

| Phase | Weight | Criteria |
| :--- | :--- | :--- |
| **Mid-Sem** | 25% | ML technique application, Feature Engineering, UI Usability, Evaluation Metrics. |
| **End-Sem** | 30% | Reasoning quality, RAG & State management implementation, Output clarity, Deployment success. |

> [!WARNING]
> Localhost-only demonstrations will **not** be accepted for final submission. Project must be hosted.
