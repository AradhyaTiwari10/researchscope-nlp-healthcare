"""
Topic Coherence Evaluation Module.

This module provides tools for quantifying topic quality and identifying the 
optimal number of topics (K) using Gensim's CoherenceModel.
"""

from gensim.models.coherencemodel import CoherenceModel
from src.topic_modeling import train_lda_model

def compute_coherence_score(model, processed_docs, dictionary, coherence='c_v'):
    """
    Computes a coherence score for an LDA model using the specified metric.
    'c_v' is common for research purposes.
    """
    coherence_model = CoherenceModel(
        model=model,
        texts=processed_docs,
        dictionary=dictionary,
        coherence=coherence
    )
    
    return coherence_model.get_coherence()

def evaluate_optimal_topic_count(processed_docs, dictionary, corpus, start=2, stop=10, step=1):
    """
    Iterates through a range of K to find the optimal number of topics based on coherence.
    Returns a dictionary mapping K to coherence scores and identifies the best K.
    """
    # Filter dictionary to ensure we only evaluate valid terms
    results = {}
    best_k = start
    max_coherence = -1
    
    for k in range(start, stop + 1, step):
        # We use fewer passes here for speed if there are many docs, 
        # but 10 is fine for university datasets
        model = train_lda_model(corpus, dictionary, num_topics=k, passes=5)
        
        score = compute_coherence_score(model, processed_docs, dictionary)
        results[k] = score
        
        if score > max_coherence:
            max_coherence = score
            best_k = k
            
    return results, best_k

if __name__ == "__main__":
    # Internal test
    from src.topic_modeling import prepare_gensim_objects
    
    sample_docs = [
        ["heart", "patient"], ["ai", "model"], ["data", "learning"],
        ["clinical", "trial"], ["healthcare", "system"], ["patient", "care"],
        ["algorithm", "diagnostic"], ["hospital", "management"],
        ["machine", "learning"], ["electronic", "health", "record"]
    ]
    
    d, c = prepare_gensim_objects(sample_docs)
    scores, opt_k = evaluate_optimal_topic_count(sample_docs, d, c, start=2, stop=3)
    
    print(f"Coherence scores: {scores}")
    print(f"Optimal topic count: {opt_k}")
