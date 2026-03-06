# -*- coding: utf-8 -*-
# stop words --> Stop words are common words that are typically filtered out in natural language processing (NLP) tasks because they do not carry significant meaning and can be considered noise in the text data. Examples of stop words include "the", "is", "in", "and", "to", etc. Removing stop words can help improve the efficiency and accuracy of NLP models by reducing the dimensionality of the text data and focusing on more meaningful words.
# speech of APJ Abdul Kalam on stop words in NLP:
paragraph = """I have three visions for India. In 3000 years of our history people from all over the world have come and invaded us, captured our lands, conquered our minds. From Alexander onwards the Greeks, the Turks, the Moguls, the Portuguese, the British, the French, the Dutch, all of them came and looted us, took over what was ours. Yet we have not done this to any other nation. We have not conquered anyone. We have not grabbed their land, their culture and their history and tried to enforce our way of life on them. Why? Because we respect the freedom of others. That is why my FIRST VISION is that of FREEDOM. I believe that India got its first vision of this in 1857, when we started the war of Independence. It is this freedom that we must protect and nurture and build on. If we are not free, no one will respect us.

We have 10 percent growth rate in most areas. Our poverty levels are falling. Our achievements are being globally recognised today. Yet we lack the self-confidence to see ourselves as a developed nation, self-reliant and self-assured. Isn't this incorrect? MY SECOND VISION for India is DEVELOPMENT. For fifty years we have been a developing nation. It is time we see ourselves as a developed nation. We are among top five nations in the world in terms of GDP.

I have a THIRD VISION. India must stand up to the world. Because I believe that unless India stands up to the world, no one will respect us. Only strength respects strength. We must be strong not only as a military power but also as an economic power. Both must go hand-in-hand. My good fortune was to have worked with three great minds. Dr.Vikram Sarabhai, of the Dept. of Space, Professor Satish Dhawan, who succeeded him and Dr. Brahm Prakash, father of nuclear material. I was lucky to have worked with all three of them closely and consider this the great opportunity of my life.

I was in Hyderabad giving this lecture, when a 14 year-old girl asked me for my autograph. I asked her what her goal in life is. She replied: I want to live in a developed India. For her, you and I will have to build this developed India. You must proclaim India is not an underdeveloped nation; it is a highly developed nation.

You say that our government is inefficient. You say that our laws are too old. You say that the municipality does not pick up the garbage. You say that the phones don't work, the railways are a joke, the airline is the worst in the world, and mails never reach their destination. You say that our country has been fed to the dogs and is the absolute pits. You say, say and say. What do you do about it?

Dear Indians, I am echoing J.F.Kennedy's words to his fellow Americans to relate to Indians ... "ASK WHAT WE CAN DO FOR INDIA AND DO WHAT HAS TO BE DONE TO MAKE INDIA WHAT AMERICA AND OTHER WESTERN COUNTRIES ARE TODAY."""

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import nltk
nltk.download('stopwords')

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
sentences = nltk.sent_tokenize(paragraph)

## Apply stop words and then apply stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)  # converting all the list of words into sentences
print(sentences)

## Now for stemming we will use the Snowball Stemmer instead of Porter Stemmer because it is more efficient and gives better results than porter stemmer. It is an improved version of the Porter Stemmer that provides better stemming results for various languages, including English.
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [snowball_stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)  # converting all the list of words into sentences
print(sentences)

#Try Lemmatization instead of stemming
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')  # Also download this for lemmatization to work properly
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [lemmatizer.lemmatize(word,pos = 'v') for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)  # converting all the list of words into sentences
print(sentences)