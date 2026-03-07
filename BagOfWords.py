#Bag of words ==> A bag of words is a representation of text that describes the occurrence of words within a document. It is a commonly used technique in natural language processing and information retrieval. The idea is to create a "bag" (or multiset) of words from a document, where the order of the words does not matter, but the frequency of each word is recorded.
from sklearn.feature_extraction.text import CountVectorizer
# Sample documents
documents = [
    "I love programming in Python",
    "Python is a great language",
    "I enjoy machine learning"
]
# Create a CountVectorizer instance
vectorizer = CountVectorizer()
# Fit and transform the documents into a bag of words representation
bag_of_words = vectorizer.fit_transform(documents)
# Get the feature names (unique words)
feature_names = vectorizer.get_feature_names_out()
# Convert the bag of words to an array and print it
print("Feature Names:", feature_names)
print("Bag of Words Representation:\n", bag_of_words.toarray())
