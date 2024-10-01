# Data science Project 1
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import svm

# Create classifier
clf = tree.DecisionTreeClassifier()
clf2 = ensemble.RandomForestClassifier()
clf3 = neural_network.MLPClassifier(max_iter=1000)
clf4 = svm.SVC()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]


Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Train the model

#clf1 - Decision Tree
clf = clf.fit(X, Y)

# Make predictions on the test set
prediction = clf.predict([[190, 70, 42]])

print(prediction)

#clf2 - Random Forest
clf2 = clf2.fit(X, Y)

prediction2 = clf2.predict([[190, 70, 42]])

print(prediction2)

#clf3 - Neural Net (Deep learning)
clf3 = clf3.fit(X, Y)

prediction3 = clf3.predict([[190, 70, 42]])

print(prediction3)

#clf4 - Support Vector Machine (SVM)
clf4 = clf4.fit(X, Y)

prediction4 = clf4.predict([[190, 70, 42]])

print(prediction4)