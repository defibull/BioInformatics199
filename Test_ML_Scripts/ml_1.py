from sklearn import tree
#feature : Weight, and texture
# smooth = 0,  bumpy = 1
# for each tool, 0 = link, 1 for non link
features = [[140, 0],[130, 0],[150, 1],[170, 1]]
# apple = 0, orange = 1
labels = [0, 0, 1, 1]
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features,labels)

print classifier.predict([[120,1]]) #try 160,1
