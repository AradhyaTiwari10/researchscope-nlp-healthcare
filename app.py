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
st.set_page_config(
    page_title="HealthRes NLP Analysis", 
    page_icon="🏥",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #fdfdfd;
    }
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.75rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1kyxreq {
        justify-content: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏥 AI & ML in Healthcare: Research Analysis System")
st.markdown("""
### Traditional NLP Intelligence Dashboard
This system provides an analytical lens into healthcare research using **strictly traditional NLP architectures**.
By avoiding Large Language Models (LLMs), we maintain deterministic reproducibility and focus on structural linguistic features (TF-IDF, LDA, and Coherence metrics).
""")

# Sidebar: Configuration
st.sidebar.header("🛠️ Pipeline Settings")
RAW_DATA_PATH = "data/raw"

# Option to load existing dataset
if st.sidebar.button("🚀 Load Research Corpus"):
    if os.path.exists(RAW_DATA_PATH):
        with st.spinner("⏳ Ingesting and assembling PDFs..."):
            documents, corpus = load_dataset(RAW_DATA_PATH)
            st.session_state['documents'] = documents
            st.session_state['corpus'] = corpus
            st.sidebar.success(f"✓ Loaded {len(documents)} papers.")
    else:
        st.sidebar.error(f"Error: {RAW_DATA_PATH} not found.")

# Main Analysis Workflow
if 'documents' in st.session_state:
    docs = st.session_state['documents']
    
    # 1. Overview metrics
    st.subheader("📊 Corpus Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents Processed", len(docs))
    m2.metric("Total Tokens", sum(len(d.split()) for d in docs))
    m3.metric("Architecture", "Traditional (v1.1)")
    
    # --- Preprocessing and Feature Extraction ---
    with st.spinner("🔄 Running spaCy/NLTK Preprocessing Pipeline..."):
        processed_docs = preprocess_corpus(docs)
    
    with st.spinner("🛰️ Extracting TF-IDF Features..."):
        vectorizer, tfidf_matrix = extract_tfidf_features(processed_docs)
        global_terms = get_global_top_terms(vectorizer, tfidf_matrix, top_n=15)
        keywords_per_doc = get_top_keywords_per_doc(vectorizer, tfidf_matrix, top_n=5)

    # --- Section: Global Keywords ---
    st.divider()
    st.header("🔑 Global Keyword Landscape")
    st.info("💡 **Traditional NLP Checkpoint:** Global scores are calculated by summing TF-IDF weights across the entire corpus.")
    
    df_global = pd.DataFrame(global_terms, columns=["Term", "Importance Score"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_global, x="Importance Score", y="Term", palette="Blues_r", ax=ax)
    plt.title("Most Significant Terms Across Corpus", fontsize=14, color="#1e293b")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    # --- Section: Topic Modeling ---
    st.divider()
    st.header("🧬 Latent Topic Analysis (LDA)")
    st.markdown("Discovering hidden semantic structures using unsupervised probabilistic modeling.")
    
    num_topics = st.select_slider("Select Number of Topics (K)", options=range(2, 11), value=5)
    
    with st.spinner("🪄 Training Generative LDA Model (Traditional)..."):
        dictionary, gensim_corpus = prepare_gensim_objects(processed_docs)
        lda_model = train_lda_model(gensim_corpus, dictionary, num_topics=num_topics)
        topics = get_topics(lda_model, num_words=10)
        coherence = compute_coherence_score(lda_model, processed_docs, dictionary)
    
    st.success(f"🎓 **Topic Coherence (C_V): {coherence:.3f}** (Optimized for traditional NLP architecture)")
    
    t_cols = st.columns(min(3, num_topics))
    for i, (tid, words) in enumerate(topics):
        with t_cols[i % 3]:
            st.markdown(f"**Topic {tid}:**")
            # Word scale for visual weight
            word_list = ", ".join([w[0] for w in words[:6]])
            st.caption(word_list)

    # --- Section: Individual Summaries ---
    st.divider()
    st.header("📝 Intelligent Document Summarization")
    st.markdown("Extracting critical sentences based on weighted sentence-level TF-IDF importance.")
    
    doc_idx = st.selectbox("Select a Research Paper to Analyze", range(len(docs)), format_func=lambda x: f"Document {x+1} - {len(docs[x].split())} words")
    
    col_sum_1, col_sum_2 = st.columns([2, 1])
    
    with col_sum_1:
        st.subheader("Extractive Summary")
        summary_text = extract_summary(docs[doc_idx], top_n=5)
        st.markdown(f'<div style="text-align: justify; background: #ffffff; padding: 20px; border-radius: 10px; border-left: 5px solid #3b82f6;">{summary_text}</div>', unsafe_allow_html=True)
        
    with col_sum_2:
        st.subheader("Key Tokens")
        st.write(", ".join(keywords_per_doc[doc_idx]))
        
        # Add a mini bar chart for this specific doc's keywords if possible
        doc_tfidf = tfidf_matrix.getrow(doc_idx).toarray().flatten()
        feat_names = vectorizer.get_feature_names_out()
        top_indices = doc_tfidf.argsort()[::-1][:10]
        doc_df = pd.DataFrame({"Term": feat_names[top_indices], "Score": doc_tfidf[top_indices]})
        st.bar_chart(doc_df.set_index("Term"))

else:
    st.info("👈 **Getting Started:** Click the sidebar button to ingest the research papers from the 'data/raw' directory.")
    st.image("https://plus.unsplash.com/premium_photo-1673953509975-576678fa6710?q=80&w=2070&auto=format&fit=crop", caption="Unlocking Healthcare Insights via NLP")
