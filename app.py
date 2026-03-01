import streamlit as st
from src.pdf_extractor import extract_text_from_pdf
from src.preprocessing import preprocess_text
from src.feature_engineering import extract_tfidf
from src.topic_modeling import perform_lda, display_topics
from src.summarizer import extractive_summary
from src.visualization import generate_wordcloud, generate_cosine_heatmap
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("AI & ML in Healthcare: Research Analysis System")

uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]
    corpus = []
    raw_texts = []
    
    for file in uploaded_files:
        raw_text = extract_text_from_pdf(file)
        raw_texts.append(raw_text)
        processed = preprocess_text(raw_text)
        corpus.append(processed)

    # D) Debug Step
    with st.expander("Debug: Parsed Corpus (First 300 words of first document)"):
        if corpus:
            st.write(" ".join(corpus[0].split()[:300]))

    # C) Extract TF-IDF
    X, vectorizer = extract_tfidf(corpus)
    
    st.subheader("Topic Modeling & Analysis")
    
    # 2. Topic Modeling (4 Topics)
    lda_model = perform_lda(X, num_topics=4)
    topics = display_topics(lda_model, vectorizer.get_feature_names_out())
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Identified Topics")
        for topic_id, words in topics:
            st.write(f"**Topic {topic_id + 1}**: {', '.join(words)}")
            
    with col2:
        st.write("### Document Topic Distribution")
        topic_dist = lda_model.transform(X)
        dominant_topic = np.argmax(topic_dist, axis=1)
        for i, name in enumerate(file_names):
            st.write(f"**{name}**: Topic {dominant_topic[i] + 1} ({int(topic_dist[i][dominant_topic[i]]*100)}% match)")

    st.divider()

    # HEATMAP SECTION (Cross-Document Analysis)
    st.subheader("Cross-Document Relationship")
    sim_matrix = cosine_similarity(X)
    fig_sim = generate_cosine_heatmap(sim_matrix, file_names)
    st.pyplot(fig_sim)

    st.divider()

    st.subheader("Visual Summary (All Papers)")
    all_text = " ".join(corpus)
    fig_wc = generate_wordcloud(all_text)
    st.pyplot(fig_wc)

    st.divider()

    st.subheader("Extractive Summary (First Paper)")
    summary = extractive_summary(raw_texts[0])
    st.write(summary)