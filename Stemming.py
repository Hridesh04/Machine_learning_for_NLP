##Stemming is a Natural Language Processing (NLP) technique that reduces words to their root or base form by removing prefixes and suffixes.
words = ["running", "runner", "ran", "easily", "fairly"]

##Porter Stemmer
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
print("Porter Stemmer results:")
for word in words:
    stemmed_word = porter_stemmer.stem(word)
    print(f"{word} -> {stemmed_word}")

#Regexp Stemmer class ---> RegexpStemmer is an NLTK stemmer that uses regular expressions to remove word suffixes based on patterns you define.
from nltk.stem import RegexpStemmer
regexp_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print("\nRegexp Stemmer results:")
for word in words:
    stemmed_word = regexp_stemmer.stem(word)
    print(f"{word} -> {stemmed_word}")

## Snowball Stemmer --> Snowball Stemmer is an improved version of the Porter Stemmer that provides better stemming results for various languages, including English.
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
print("\nSnowball Stemmer results:")
for word in words:
    stemmed_word = snowball_stemmer.stem(word)
    print(f"{word} -> {stemmed_word}")

print(snowball_stemmer.stem("fairly")),
print(snowball_stemmer.stem("sportingly"))
