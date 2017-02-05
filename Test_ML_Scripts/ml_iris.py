from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pprint
import warnings



pp = pprint.PrettyPrinter(indent=4)
iris = load_iris()
test_idx = [0,50,100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

test_target, test_data = iris.target[test_idx], iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# print test_target
print iris.data[50]
print clf.predict(iris.data[50])


# print iris.feature_names
# print iris.target_names
# pp.pprint(iris.data)
# pp.pprint(iris.target)
