# Milestone 1: Traditional NLP Research Analysis System
## Topic: Artificial Intelligence and Machine Learning in Healthcare

### 1. Methodology Overview
This system implements a fully traditional (non-generative) NLP pipeline for the structural analysis of academic research papers. The architecture is designed to perform document ingestion, preprocessing, feature extraction, and topic modeling without the use of Large Language Models (LLMs) or Transformers.

### 2. Pipeline Implementation
*   **Data Ingestion**: Page-level text extraction from scientific PDFs using `pdfplumber`.
*   **Preprocessing**: Lemmatization via spaCy's `en_core_web_sm`, token filtering (stopwards, alpha-only, length > 2), and specialized regex cleaning for academic noise (PMIDs, DOIs, URLs).
*   **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) to identify both document-local keywords and global corpus-wide significance.
*   **Topic Modeling**: Latent Dirichlet Allocation (LDA) via Gensim to extract semantic themes.
*   **Summarization**: Sentence ranking based on intra-document TF-IDF weights (Extractive approach).

### 3. Critical Technical Observations (For Viva)
Durante the development and evaluation of this system, several key technical characteristics of traditional NLP were observed:

| Observation | Technical Explanation |
| :--- | :--- |
| **Coherence Score (~0.51)** | Scientific text exhibits high "semantic density." For specialized healthcare papers, a score in the 0.45–0.55 range is typical for traditional LDA as the model balances broad terms (e.g., *healthcare*, *patient*) with specific medical nuances. |
| **Topic Overlap** | High overlap is expected in domain-specific corpora. Since all papers discuss AI in Healthcare, terms like "data" and "care" naturally appear across multiple topics, reflecting the interconnected nature of the research. |
| **Extractive Summarization** | Traditional TF-IDF summarization can sometimes pull sentences with citation artifacts (e.g., "Author et al."). While our pipeline cleans common patterns, the structural complexity of scientific PDFs remains a challenge for rule-based tokenizers. |
| **PDF Extraction Noise** | We use `pdfplumber` instead of basic `PyPDF2` to maintain document flow. However, scientific papers' multi-column layouts frequently introduce line-break artifacts that traditional sentence tokenizers must resolve. |

### 4. Conclusion
By strictly adhering to traditional algorithms (TF-IDF, LDA), this milestone provides a baseline for research analysis that is transparent, computationally efficient, and fully reproducible without reliance on external black-box APIs.
