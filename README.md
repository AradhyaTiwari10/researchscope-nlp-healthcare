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
│   ├── raw/              ← PDF research papers
│   └── processed/        ← Intermediate outputs
│
├── src/
│   ├── data_loader.py       ← PDF ingestion & corpus assembly
│   ├── preprocessing.py     ← Text cleaning, tokenization, lemmatization, bigrams
│   ├── feature_extraction.py← TF-IDF feature extraction
│   ├── topic_modeling.py    ← LDA topic modeling (Gensim)
│   ├── evaluation.py        ← Topic coherence evaluation
│   └── summarization.py     ← Extractive summarization (TF-IDF sentence scoring)
│
├── notebooks/
│   └── experiments.ipynb
│
├── app.py                ← Streamlit UI
├── requirements.txt
└── README.md
```

---

## Dependencies

```
pdfplumber, nltk, spacy, scikit-learn, gensim, pandas, numpy, matplotlib, seaborn, streamlit
```

Install all with:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Running the Application

```bash
streamlit run app.py
```

Then click **"Load Research Corpus"** in the sidebar.

---

## Limitations

This system is a **Milestone 1 prototype** using traditional NLP only. The following known limitations apply:

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