import numpy as np
from numpy import *
import nltk
from nltk.stem.porter import PorterStemmer

Stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)
    
def stem(word):
    return Stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence,words):
    sentence_word = [stem(word) for word in tokenize_sentence]
    bag = np.zeros(len(words),dtype=np.float32)

    for idx , w in enumerate(words):
        if w in sentence_word:
            bag[idx] = 1

    return bag