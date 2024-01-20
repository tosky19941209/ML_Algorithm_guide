import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target
print(len(X))

x_train, x_test, y_train, y_test = train_test_split(X, y , test_size = 0.2, random_state = 42)

clf= DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))