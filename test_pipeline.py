"""
test_pipeline.py
-----------------
Automated test script for the refactored NLP pipeline.
Phase 0 – ResearchScope NLP Healthcare
"""

from src.preprocessing import preprocess
from src.feature_engineering import extract_features
from src.topic_modeling import get_topics
from src.similarity import compute_similarity
from src.summarizer import summarize
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample texts representing medical research abstracts
SAMPLE_TEXTS = [
    """
    ABSTRACT
    Machine learning models, specifically deep neural networks, have shown immense potential 
    in healthcare for predictive analytics. This study evaluates a convolutional neural network 
    approach to classify radiological images. Our methodology demonstrates a 95% accuracy 
    in early detection of lung cancer. These findings suggest that AI can significantly 
    improve clinical diagnostic workflows.
    """,
    """
    ABSTRACT
    This paper discusses the application of Support Vector Machines (SVM) in clinical 
    diagnostics. We propose a new feature extraction methodology for analyzing patient 
    electronic health records (EHR). The dataset consists of 5000 patient records. 
    The results show an improvement over traditional statistical methods for predicting 
    cardiovascular diseases.
    """,
    """
    ABSTRACT
    Natural Language Processing (NLP) provides a novel approach to automate the extraction 
    of critical patient data from unstructured EHR narratives. In this research, we introduce 
    a transformer-based architecture that identifies medication discrepancies. The system 
    was validated on a large clinical dataset, demonstrating high recall.
    """
]

def main():
    print("🚀 Starting NLP Pipeline Test...\n")
    
    # 1. Preprocessing
    print("1️⃣ Testing Preprocessing...")
    corpus = []
    for i, text in enumerate(SAMPLE_TEXTS):
        cleaned = preprocess(text)
        corpus.append(cleaned)
        print(f"  [Original  {i+1}] {text.strip()[:60]}...")
        print(f"  [Processed {i+1}] {cleaned[:60]}...\n")
        
    # 2. Feature Engineering
    print("2️⃣ Testing Feature Engineering (TF-IDF)...")
    X, vectorizer = extract_features(corpus)
    print(f"  => Feature matrix shape: {X.shape}")
    print(f"  => Vocabulary size: {len(vectorizer.vocabulary_)}\n")
    
    # 3. Topic Modeling
    print("3️⃣ Testing Topic Modeling (LDA)...")
    topics = get_topics(X, vectorizer, num_topics=2, num_words=5)
    for topic_idx, top_words in topics:
        print(f"  => Topic {topic_idx + 1}: {', '.join(top_words)}")
    print()

    # 4. Similarity Calculation
    print("4️⃣ Testing Similarity mapping...")
    sim_matrix = compute_similarity(X)
    print(f"  => Similarity Matrix (Shape: {sim_matrix.shape}):")
    for row in sim_matrix:
        print(f"     {[round(val, 2) for val in row]}")
    print()

    # 5. Summarization
    print("5️⃣ Testing Summarization...")
    # Instantiate a fresh vectorizer for the summarizer as per its requirements
    summarizer_vectorizer = TfidfVectorizer()
    for i, text in enumerate(SAMPLE_TEXTS):
        summary = summarize(text, summarizer_vectorizer, num_sentences=2)
        print(f"  => Summary {i+1}:\n     {summary}\n")
        
    print("✅ Pipeline test completed successfully.")

if __name__ == "__main__":
    main()
