from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf(corpus):
    vectorizer = TfidfVectorizer(
        max_features=2500,
        min_df=2,
        max_df=0.85,
        ngram_range=(1,2)
    )
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer