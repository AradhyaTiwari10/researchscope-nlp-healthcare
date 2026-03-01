"""
LDA Topic Modeling Engine.

This module uses Gensim's Latent Dirichlet Allocation (LDA) implementation
to discover latent topics within the document collection.
"""

from gensim.corpora import Dictionary
from gensim.models import LdaModel

def prepare_gensim_objects(processed_docs):
    """
    Creates a Gensim Dictionary and Bag-of-Words (BoW) corpus from tokenized documents.
    """
    dictionary = Dictionary(processed_docs)
    # Filter extremes: keep tokens appearing in >= 2 docs and in < 70% of docs
    dictionary.filter_extremes(no_below=2, no_above=0.7)
    
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    
    return dictionary, corpus

def train_lda_model(corpus, dictionary, num_topics=5, passes=10, random_state=42):
    """
    Trains an LDA model on the BoW corpus.
    """
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        passes=passes,
        random_state=random_state,
        alpha='auto'
    )
    
    return lda_model

def get_topics(lda_model, num_words=10):
    """
    Extracts the top N terms for each identified topic.
    Returns a list of topics, each represented as a list of (word, weight).
    """
    # Extract [topic_id, list of words/weights]
    topics = lda_model.show_topics(num_topics=-1, num_words=num_words, formatted=False)
    
    return topics

def get_doc_topics(lda_model, corpus):
    """
    Maps each document to its dominant topic and its probability.
    """
    doc_topics = []
    for bow in corpus:
        # returns [(topic_id, prob), (topic_id, prob), ...]
        topics_probs = sorted(lda_model.get_document_topics(bow), key=lambda x: x[1], reverse=True)
        doc_topics.append(topics_probs[0] if topics_probs else (None, 0.0))
        
    return doc_topics

if __name__ == "__main__":
    # Internal test
    sample_docs = [
        ["heart", "disease", "diagnosis", "healthcare", "patient"],
        ["machine", "learning", "data", "prediction", "model"],
        ["heart", "surgery", "patient", "recovery", "care"],
        ["deep", "learning", "neural", "network", "image"],
        ["clinical", "trial", "drug", "fda", "patient"]
    ]
    
    d, c = prepare_gensim_objects(sample_docs)
    model = train_lda_model(c, d, num_topics=2)
    topics = get_topics(model)
    
    for tid, words in topics:
        print(f"Topic {tid}: {[w[0] for w in words]}")
