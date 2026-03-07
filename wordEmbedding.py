###   word embedding --> Word embedding is a technique in natural language processing (NLP) that represents words as dense vectors in a continuous vector space. 
#These vectors capture semantic relationships between words, allowing NLP models to understand and process language more effectively. 
#Word embeddings are typically learned from large corpora of text using algorithms like Word2Vec, GloVe, or FastText, and they enable tasks such as sentiment analysis, machine translation, and text classification by providing a meaningful representation of words in a way that captures their contextual meaning.
###   in NLP word embedding is a term used for the representation of words for the text analysis,
#typically in the form of a real valued vectors that encodes the meaning of the words such that the words
#that are similar in meaning are close to each other in the vector space.



### Word2vec is a technique for natural language processing published by google in 2013,
# it is a two layer neural network that takes a text corpus as input and produces a vector space, typically of several hundred dimensions, with each unique word in the corpus being assigned a corresponding vector in the space.
# the word2vec a lgorithm uses a neural network model to learn associations form a large corpus of the text.
# once,trained such a model can detect synonymous words or suggest additional words for a partial sentence.
# as a name implies word2vec represents each distinct word with a particular list of numbers called a vector.

#The vector is a list of floating point numbers that represent the word in a high dimensional space. The position of a word vector in the space is learned from the text and is based on the words that appear around it in the text. Words that share common contexts in the text are located close to one another in the space.
