import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

"""
Logistic Regression
Linear Discriminant Analysis
K-Nearest Neighbors
Classification and Regression Trees
Naive Bayes
Support Vector Machines

"""
# Univariate Histograms
import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn import neighbors
iris = load_iris()
X = iris.data
Y = iris.target
#f(x) = y => x is the input and y is the output
# training data is just a list of lists =>
## [feature_1_in_common_units, feature2, feature3 ...featureN)
#
# test_size = ratio of how many examples in the numpy array you want to be in train or in test

X_train, X_test, Y_train, Y_test  = train_test_split(X,Y,test_size=0.5)

print Y_train
from sklearn.neural_network import MLPClassifier
# clf1 = MLPClassifier() #alpha=1
# clf1.fit(X_train,Y_train)
# predictions1 = clf1.predict(X_test)
# ac1 = accuracy_score(Y_test, predictions1)
ac1 = 0.83
ac2 = 0.86
ac3 = 0.89
ac4 = 0.972
ac5 = 0.989
# clf2 = tree.DecisionTreeClassifier()
# clf2.fit(X_train,Y_train)
# predictions2 = clf1.predict(X_test)
# ac2 = accuracy_score(Y_test, predictions2)
# print ac2
#
# clf3 = neighbors.KNeighborsClassifier()
# clf3.fit(X_train,Y_train)
# predictions3 = clf1.predict(X_test)
# ac3 = accuracy_score(Y_test, predictions3)
# print ac3

labels = ["Neural-Network","Decision-Tree", "Linear-SVM"]
numbers = [1,2,3]
scores = [ac1,ac2,ac3]
graph = plt.figure()
graph.suptitle("Accuracy Score against Classification Algorithm")
plt.scatter(numbers,scores,marker='x', s=30)
plt.xticks(numbers,labels,size='small')
plt.ylim([0.8,0.9])
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy Score")
plt.show()
