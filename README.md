# Project 5: Customer Churn Prediction & Agentic Retention Strategy

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

### 📁 Project Structure

```
researchscope-nlp-healthcare/
│
├── data/
│   └── (User must drop 10 PDF research papers here to upload via UI)
├── app.py                     # Streamlit Main UI Application
├── feature_engineering.py     # TF-IDF Feature Extraction logic
├── pdf_extractor.py           # Heuristic Abstract/Intro PDF parser
├── preprocessing.py           # NLP cleaning, Tokenization, Lemmatization, Stopwords
├── summarizer.py              # TF-IDF sentence scoring for Extractive Summaries
├── topic_modeling.py          # LDA Topic Modeling implementation
├── visualization.py           # WordCloud rendering pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation (You are here)
```

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