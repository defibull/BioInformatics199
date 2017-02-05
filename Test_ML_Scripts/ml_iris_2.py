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


iris = load_iris()
X = iris.data
Y = iris.target
#f(x) = y => x is the input and y is the output
# training data is just a list of lists =>
## [feature_1_in_common_units, feature2, feature3 ...featureN)
#
# test_size = ratio of how many examples in the numpy array you want to be in train or in test

X_train, X_test, Y_train, Y_test  = train_test_split(X,Y,test_size=0.5)
# from sklearn import tree
# clf = tree.DecisionTreeClassifier()
# from sklearn import neighbors
# clf = neighbors.KNeighborsClassifier()
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier() #alpha=1
clf.fit(X_train,Y_train)
predictions = clf.predict(X_test)
print accuracy_score(Y_test, predictions)
