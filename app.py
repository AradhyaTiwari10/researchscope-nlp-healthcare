import streamlit as st
from pdf_extractor import extract_text_from_pdf
from preprocessing import preprocess_text
from feature_engineering import extract_tfidf
from topic_modeling import perform_lda, display_topics
from summarizer import extractive_summary
from visualization import generate_wordcloud

st.title("AI & ML in Healthcare: Research Analysis System")

uploaded_files = st.file_uploader(
    "Upload Research Papers (PDF)",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    corpus = []
    raw_texts = []
    
    for file in uploaded_files:
        raw_text = extract_text_from_pdf(file)
        raw_texts.append(raw_text)
        processed = preprocess_text(raw_text)
        corpus.append(processed)

    X, vectorizer = extract_tfidf(corpus)
    lda_model = perform_lda(X, num_topics=5)

    topics = display_topics(
        lda_model,
        vectorizer.get_feature_names_out()
    )

    st.subheader("📌 Identified Topics")
    for topic_id, words in topics:
        st.write(f"Topic {topic_id}: {', '.join(words)}")

    st.subheader("📊 WordCloud (All Papers)")
    all_text = " ".join(corpus)
    fig = generate_wordcloud(all_text)
    st.pyplot(fig)

    st.subheader("📄 Extractive Summary (First Paper)")
    summary = extractive_summary(raw_texts[0])
    st.write(summary)