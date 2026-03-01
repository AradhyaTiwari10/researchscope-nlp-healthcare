import nltk
import spacy
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    
    tokens = [
        word for word in tokens
        if word not in stop_words
        and word not in string.punctuation
        and word.isalpha()
    ]
    
    doc = nlp(" ".join(tokens))
    lemmatized = [token.lemma_ for token in doc]
    
    return " ".join(lemmatized)