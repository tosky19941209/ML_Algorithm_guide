import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

np.random.seed(10)
X = np.random.rand(100,1)
y = 2 * X + 3 + np.random.rand(100,1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 , random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

plt.scatter(X_train, y_train, color='red', label='training data')
plt.scatter(X_test, y_test, color='blue', label='tested data')
plt.plot(X, lr.predict(X), color='green', label='regression line')
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

