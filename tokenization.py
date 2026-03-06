import nltk
import ssl

# Fix SSL certificate issues if present
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_unverified_https_context = _create_unverified_https_context

# Download with verification
result = nltk.download('punkt_tab')
if not result:
    print("Download failed. Trying alternative 'punkt'...")
    nltk.download('punkt')

from nltk.tokenize import sent_tokenize

corpus = """Hello welcome to Krish Naik's NLP tutorials.
My name is Dear!
He is a data scientist and a machine learning engineer.
"""

# Tokenization 
# Tokenization is the process of breaking down a large paragraph into smaller pieces called tokens.
# Sentence Tokenization (Paragraph to sentence)

sentences = sent_tokenize(corpus)

# Display results
for i, sentence in enumerate(sentences, 1):
    print(f"{i}. {sentence}")

# we want to break down the sentences into words, we can use word_tokenize
from nltk.tokenize import word_tokenize
# Word Tokenization (Sentence to word)
for i, sentence in enumerate(sentences, 1):
    words = word_tokenize(sentence)
    print(f"Sentence {i} tokens: {words}")

# print each word seperatly
print("\nAll words in corpus (one per line):")
all_words = word_tokenize(corpus)
for word in all_words:
    print(word)

# now we will remove ' and other character with the help of treebankwordTokenizer
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
treebank_tokens = tokenizer.tokenize(corpus)

print("\nTreebank Word Tokenizer results:")
for i, token in enumerate(treebank_tokens, 1):
    print(f"{i}. {token}")
