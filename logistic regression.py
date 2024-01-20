import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 2)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_test = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test,y_test)

print("Target of Y ", y)
print("Y-test ", y_test)
print(accuracy)
