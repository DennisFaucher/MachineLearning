import nltk
#import pandas
import pickle

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

import codecs

# load the model from disk
filename = '1MTweets_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

phrase = raw_input("Please type a phrase: ")
print(phrase)

while phrase != "quit":
    result = loaded_model.classify(format_sentence(phrase))
    print(result)
    phrase = raw_input("Please type a phrase: ")
    print phrase
