import streamlit as st
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import preprocess_text
from src.feature_engineering import extract_tfidf
from src.topic_modeling import perform_lda, display_topics
from src.summarizer import extractive_summary
from src.visualization import generate_wordcloud, generate_cosine_heatmap
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page Configuration
st.set_page_config(page_title="Healthcare Research AI", layout="wide")

st.title("AI & ML in Healthcare: Research Analysis System")

# 1. Sidebar explanation
with st.sidebar:
    st.header("Project Overview")
    st.info("""
    This system uses **Classical NLP** (no LLMs) to analyze healthcare research. 
    It identifies key topics, evaluates document similarity, and generates summaries.
    """)
    st.markdown("---")
    st.markdown("**Milestone 1:** Topic Modeling & Text Analytics")

# 2. File Upload
uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    corpus = []
    raw_texts = []
    
    # Progress Bar for Processing
    with st.status("Preprocessing PDFs...", expanded=True) as status:
        for file in uploaded_files:
            raw_text = extract_text_from_pdf(file)
            raw_texts.append(raw_text)
            processed = preprocess_text(raw_text)
            corpus.append(processed)
        status.update(label="Preprocessing Complete!", state="complete", expanded=False)

    # A) Feature Engineering (TF-IDF)
    X, vectorizer = extract_tfidf(corpus)
    
    # B) Topic Modeling (LDA)
    st.header("1. Topic Discovery (LDA)")
    with st.expander("ℹ️ What is this?"):
        st.write("""
        **Latent Dirichlet Allocation (LDA)** is a statistical model that identifies clusters of related words called 'Topics'. 
        By looking at which words appear together, we can see the primary themes (e.g., Genomics vs Clinical Trials) 
        without reading every page.
        """)

    lda_model = perform_lda(X, num_topics=4)
    topics = display_topics(lda_model, vectorizer.get_feature_names_out())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Thematic Clusters")
        for topic_id, words in topics:
            st.write(f"**Topic {topic_id + 1}**: {', '.join(words)}")
            
    with col2:
        st.write("### Document Mapping")
        topic_dist = lda_model.transform(X)
        dominant_topic = np.argmax(topic_dist, axis=1)
        for i, name in enumerate(file_names):
            st.write(f"**{name}**: Topic {dominant_topic[i] + 1} ({int(topic_dist[i][dominant_topic[i]]*100)}% reliability)")

    st.divider()

    # C) Relationship Mapping (Cosine Similarity)
    st.header("2. Cross-Document Similarity")
    with st.expander("ℹ️ What is this?"):
        st.write("""
        **Cosine Similarity** measures the 'mathematical distance' between documents in terms of vocabulary. 
        A score of **1.0 (Dark Blue)** means the papers are almost identical in their research focus, 
        making it easy to spot redundant reports or related studies.
        """)
    sim_matrix = cosine_similarity(X)
    fig_sim = generate_cosine_heatmap(sim_matrix, file_names)
    st.pyplot(fig_sim)

    st.divider()

    # D) Visual Word Analysis
    st.header("3. Keyword Cloud")
    with st.expander("ℹ️ What is this?"):
        st.write("""
        This **WordCloud** highlights the most significant terms across your entire research corpus. 
        Larger words appeared more frequently in the 'cleaned' dataset, providing a quick visual 
        Exploratory Data Analysis (EDA).
        """)
    all_text = " ".join(corpus)
    fig_wc = generate_wordcloud(all_text)
    st.pyplot(fig_wc)

    st.divider()

    # E) Interactive Summary Engine
    st.header("4. Smart Extractive Summarizer")
    with st.expander("ℹ️ What is this?"):
        st.write("""
        This tool uses **TF-IDF ranking** to find the most 'content-rich' sentences in the Abstract. 
        It re-sequences them to give you a clean, 3-sentence summary of the research methodology 
        and findings without the scientific noise.
        """)
    
    selected_doc = st.selectbox("Select a Research Paper to Summarize:", file_names)
    selected_index = file_names.index(selected_doc)
    
    with st.chat_message("assistant"):
        summary = extractive_summary(raw_texts[selected_index])
        st.write(f"**Summary of {selected_doc}:**")
        st.write(summary)

    # Debug Section (Footer)
    st.markdown("---")
    with st.expander("Developer Debugging info"):
        st.write(f"**Total Documents:** {len(corpus)}")
        st.write(f"**TF-IDF Matrix Shape:** {X.shape}")
        if corpus:
            st.write("**Snippet of Cleaned Text (Doc 1):**")
            st.write(" ".join(corpus[0].split()[:200]) + "...")