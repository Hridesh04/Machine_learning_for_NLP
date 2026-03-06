##Lemmatization is the process of reducing a word to its base or root form, known as a lemma.\n
# It is a common text preprocessing technique used in natural language processing (NLP) to normalize words and improve the performance of various NLP tasks.
## the main difference between stemming and lemmatization is that stemming simply chops off the ends of words, while lemmatization considers the context and meaning of the word to determine its base form.
## Lemmatization typically produces more accurate results than stemming, as it takes into account the grammatical structure and semantics of the word, whereas stemming can sometimes produce non-existent or incorrect base forms.
## For example, the word "running" would be stemmed to "run" using stemming, but lemmatization would also consider the context and return "run" as the lemma, which is the correct base form of the word.


words = ["running", "runner", "ran", "easily", "fairly"]

import nltk
nltk.download('omw-1.4')  # Also download this for lemmatization to work properly
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

words = ["running", "runner", "ran", "easily", "fairly"]
lemmatizer = WordNetLemmatizer()

for word in words:
    print(f"{word} -> {lemmatizer.lemmatize(word,pos = 'v')}")



##NLTK provides WordNetLemmatizer class which is thin wrapper around the wordnet corpus.\n
##this class uses morphy function of wordnet to find the lemma of a word. It takes two parameters, 
#the word to be lemmatized and the part of speech (POS) tag of the word. The POS tag is important 
#because it helps the lemmatizer to determine the correct lemma for a given word based on its context in a sentence.
# it can be used for chabot,test summarization and Q&A system to understand the meaning of the words and provide accurate responses. 
# For example, if a user asks "What is the running time of the movie?", the lemmatizer can help identify that "running" is a verb and lemmatize it to "run", 
# which can then be used to retrieve relevant information about the movie's runtime.