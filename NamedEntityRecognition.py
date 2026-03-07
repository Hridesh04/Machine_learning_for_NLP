##Named Entity Recognition (NER) is a subtask of Natural Language Processing (NLP) that focuses on identifying and classifying named entities in text into predefined categories such as person names, organizations, locations, dates, etc. NER is crucial for various applications like information extraction, question answering, and machine translation.
import nltk


# Download required resources FIRST
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('punkt')




sentence = "Apple is looking at buying U.K. startup for $1 billion"

words = nltk.word_tokenize(sentence)
tag_elements = nltk.pos_tag(words)

print(tag_elements)

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.ne_chunk(tag_elements).draw()
