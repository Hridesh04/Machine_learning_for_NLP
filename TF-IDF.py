#TF-IDF is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus). It stands for Term Frequency-Inverse Document Frequency. The TF-IDF score is calculated by multiplying two components:
#1. Term Frequency (TF): This measures how frequently a term appears in a document. It is calculated as the number of times a term appears in a document divided by the total number of terms in that document.
#2. Inverse Document Frequency (IDF): This measures how important a term is in the entire corpus. It is calculated as the logarithm of the total number of documents divided by the number of documents containing the term.
from sklearn.feature_extraction.text import TfidfVectorizer
# Sample documents
documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy machine learning"
]
# Create a TfidfVectorizer instance amd also implement with max_features
vectorizer = TfidfVectorizer(max_features=10)
# Fit and transform the documents into a TF-IDF representation
tfidf_matrix = vectorizer.fit_transform(documents)
# Get the feature names (unique words)
feature_names = vectorizer.get_feature_names_out()
# Convert the TF-IDF matrix to an array and print it
print("Feature Names:", feature_names)
print("TF-IDF Representation:\n", tfidf_matrix.toarray())

