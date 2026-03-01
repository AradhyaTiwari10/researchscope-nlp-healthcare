from sklearn.decomposition import LatentDirichletAllocation

def perform_lda(X, num_topics=5):
    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(X)
    return lda

def display_topics(model, feature_names, num_words=10):
    topics = []
    for idx, topic in enumerate(model.components_):
        top_words = [
            feature_names[i]
            for i in topic.argsort()[:-num_words - 1:-1]
        ]
        topics.append((idx, top_words))
    return topics