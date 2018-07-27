import random
random.seed(18)
# total data point
#n_points = 100
# data points
n_points = 296
# 624 rows in SFDC_NoOpp
X = [(random.random(), random.random()) for ii in range(0,n_points)]
# data labels
y = [round(random.random()) for ii in range(0,n_points)]
# split into train/test sets
split = int(0.75*n_points) # Train:Test::0.75:0.25

import numpy as np
fname = 'character-deaths-clean-nohead.csv'

#Test1 - X=Allegiance, Y=Chapters, y=Nobility - 69% accurate
#X = np.loadtxt(fname, dtype=np.int, delimiter=',', usecols=(2,6))
# Use Columns 'Allegiance Num', 'Death Total Chapters' to plot the data points. Skip header columns.
# Start counting from 0 dummy
# y = np.loadtxt(fname, dtype=np.str, delimiter=',', usecols=(13))
# Use Column 'Nobility' as the classifier to group the data

#Test2 - X=Allegiance, Y=Nobility, y=Chapters - 3% accurate
#X = np.loadtxt(fname, dtype=np.int, delimiter=',', usecols=(2,13))
##Use Columns 'Allegiance Num', 'Nobility' to plot the data points. Skip header columns.
## Start counting from 0 dummy
#y = np.loadtxt(fname, dtype=np.str, delimiter=',', usecols=(6))
## Use Column 'Death Total Chapters' as the classifier to group the data

#Test3 - X=Allegiance, Y=Intro Book, y=Chapters - 8% accurate
#X = np.loadtxt(fname, dtype=np.int, delimiter=',', usecols=(2,9))
##Use Columns 'Allegiance Num', 'Nobility' to plot the data points. Skip header columns.
## Start counting from 0 dummy
#y = np.loadtxt(fname, dtype=np.str, delimiter=',', usecols=(6))
## Use Column 'Death Total Chapters' as the classifier to group the data

#Test4 - X=Allegiance, Y=Serial Intro Chapter, y=Chapters - 18% accurate
#X = np.loadtxt(fname, dtype=np.int, delimiter=',', usecols=(2,10))
##Use Columns 'Allegiance Num', 'Nobility' to plot the data points. Skip header columns.
## Start counting from 0 dummy
#y = np.loadtxt(fname, dtype=np.str, delimiter=',', usecols=(6))
## Use Column 'Death Total Chapters' as the classifier to group the data

#Test5 - X=Allegiance, Y=Nobility, y=Death Year - 22% accurate
X = np.loadtxt(fname, dtype=np.int, delimiter=',', usecols=(2,13))
##Use Columns 'Allegiance Num', 'Nobility' to plot the data points. Skip header columns.
## Start counting from 0 dummy
y = np.loadtxt(fname, dtype=np.str, delimiter=',', usecols=(3))
## Use Column 'Death Total Chapters' as the classifier to group the data


X_train = X[0:split] # features_train
X_test  = X[split:] # features_test
y_train = y[0:split] # labels_train
y_test  = y[split:] # labels_test
# import Gaussian Naive Bayes (GaussianNB)
from sklearn.naive_bayes import GaussianNB
# define classifier
clf = GaussianNB()
# fit the training data features and it's labels
clf.fit(X_train, y_train)
# predict labels for the test dataset
pred = clf.predict(X_test)

##from sklearn.metrics import accuracy_score
##print(accuracy_score(pred, y_test))
#sklearn module accuracy - 21%

count = len(['matched' for idx, label in enumerate(y_test) if label == pred[idx]])
print("Your GOT nobility longevity learning prediction is accurate to"),; print('%.2f' % (float(count) / len(y_test))),; print("%")
# accuracy without importing module - 21%

#Print a prediction
#print"I predict the nobility of is ",; print(clf.predict([[4,90]]))


#Old machine learning
##x,y = xdata[:-10], ydata[:-10]
#Learn on all data except last 10 rows
##from sklearn.svm import SVC
##clf = SVC(gamma=0.001)
##clf.fit(x, y)
##print"I predict Shane's 90 day opp stage is ",; print(clf.predict([[4,90]]))
#Predict the stage of one of Frank's opps that is 100 days old
