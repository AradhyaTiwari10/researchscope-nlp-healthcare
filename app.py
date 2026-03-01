"""
AI in Healthcare: Research Analysis System (Main Application)

This UI integrates the entire traditional NLP pipeline from document loading 
to summarization and topic modeling for research-driven healthcare analysis.
"""

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from src.data_loader import load_dataset
from src.preprocessing import preprocess_corpus
from src.feature_extraction import extract_tfidf_features, get_top_keywords_per_doc, get_global_top_terms
from src.topic_modeling import prepare_gensim_objects, train_lda_model, get_topics, get_doc_topics
from src.evaluation import compute_coherence_score
from src.summarization import extract_summary

# Set UI Configuration
st.set_page_config(page_title="HealthRes NLP Analysis", layout="wide")

st.title("🏥 AI & ML in Healthcare: Research Analysis System")
st.markdown("""
Welcome to the research analysis system. This dashboard uses **traditional NLP only** (NLTK, spaCy, scikit-learn, and Gensim) 
to analyze scientific papers on AI-driven healthcare without using LLMs or Transformers.
""")

# Sidebar: Configuration
st.sidebar.header("Pipeline Configuration")
RAW_DATA_PATH = "data/raw"

# Option to load existing dataset
if st.sidebar.button("🚀 Load Local Dataset"):
    if os.path.exists(RAW_DATA_PATH):
        with st.spinner("⏳ Ingesting PDFs..."):
            documents, corpus = load_dataset(RAW_DATA_PATH)
            st.session_state['documents'] = documents
            st.session_state['corpus'] = corpus
            st.success(f"Successfully loaded {len(documents)} documents.")
    else:
        st.error(f"Path '{RAW_DATA_PATH}' does not exist.")

# Main Analysis Workflow
if 'documents' in st.session_state:
    docs = st.session_state['documents']
    
    # 1. Overview metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents Analysis", len(docs))
    col2.metric("Total Tokens (Approx)", sum(len(d.split()) for d in docs))
    
    # --- Preprocessing and Feature Extraction ---
    with st.spinner("🔄 Running Preprocessing Pipeline..."):
        processed_docs = preprocess_corpus(docs)
    
    with st.spinner("🛰️ Extracting TF-IDF Features..."):
        vectorizer, tfidf_matrix = extract_tfidf_features(processed_docs)
        global_terms = get_global_top_terms(vectorizer, tfidf_matrix, top_n=15)
        keywords_per_doc = get_top_keywords_per_doc(vectorizer, tfidf_matrix, top_n=5)

    # --- Section: Global Keywords ---
    st.header("🔑 Global Keyword Landscape")
    df_global = pd.DataFrame(global_terms, columns=["Term", "Mean TF-IDF"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_global, x="Mean TF-IDF", y="Term", palette="viridis", ax=ax)
    st.pyplot(fig)

    # --- Section: Topic Modeling ---
    st.header("🧬 Latent Topic Analysis")
    num_topics = st.slider("Select Number of Topics (K)", 2, 10, 5)
    
    with st.spinner("🪄 Training LDA Model..."):
        dictionary, gensim_corpus = prepare_gensim_objects(processed_docs)
        lda_model = train_lda_model(gensim_corpus, dictionary, num_topics=num_topics)
        topics = get_topics(lda_model, num_words=10)
        coherence = compute_coherence_score(lda_model, processed_docs, dictionary)
    
    st.info(f"✨ Model Coherence Score (C_V): **{coherence:.3f}**")
    
    t_cols = st.columns(min(3, num_topics))
    for i, (tid, words) in enumerate(topics):
        with t_cols[i % 3]:
            st.subheader(f"Topic {tid}")
            st.markdown(", ".join([w[0] for w in words]))

    # --- Section: Individual Summaries ---
    st.header("📝 Individual Research Summaries")
    doc_idx = st.selectbox("Select a Research Paper to Summarize", range(len(docs)), format_func=lambda x: f"Document {x+1}")
    
    col_sum_1, col_sum_2 = st.columns([2, 1])
    
    with col_sum_1:
        st.subheader("Extractive Summary")
        summary_text = extract_summary(docs[doc_idx], top_n=5)
        st.write(summary_text)
        
    with col_sum_2:
        st.subheader("Top Features")
        st.write(keywords_per_doc[doc_idx])
else:
    st.write("👈 Click 'Load Local Dataset' in the sidebar to begin analysis.")
