from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=1500)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer