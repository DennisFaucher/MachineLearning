import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

#print(format_sentence("The cat is very cute"))

import codecs

pos = []
#with open("./pos.txt") as f:
#f = codecs.open('./140pos.csv', encoding='utf-8')
f = codecs.open('./140pos.csv', encoding='latin-1')
for i in f:
        pos.append([format_sentence(i), 'pos'])

neg = []
#with open("./neg.txt") as f:
#f = codecs.open('./140neg.csv', encoding='utf-8')
f = codecs.open('./140neg.csv', encoding='latin-1')
for i in f:
        neg.append([format_sentence(i), 'neg'])

# next, split labeled data into the training and test data
training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

from nltk.classify import NaiveBayesClassifier
classifier = NaiveBayesClassifier.train(training)
#classifier.show_most_informative_features()

example1 = "I love asparagus"
print(classifier.classify(format_sentence(example1)))

# Compute accuracy of model
# 0.751961342808
#from nltk.classify.util import accuracy
#print(accuracy(classifier, test))
