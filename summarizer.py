import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("punkt")

def extractive_summary(text, num_sentences=3):
    sentences = sent_tokenize(text)
    
    if len(sentences) <= num_sentences:
        return text
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)
    
    sentence_scores = X.sum(axis=1)
    
    ranked = sorted(
        [(sentence_scores[i, 0], s)
         for i, s in enumerate(sentences)],
        reverse=True
    )
    
    summary = " ".join([s for _, s in ranked[:num_sentences]])
    return summary