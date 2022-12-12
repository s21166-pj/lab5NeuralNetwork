import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import genfromtxt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

"""
Authors:
- Badysiak Pawel - s21166
- Turek Wojciech - s21611
"""

# data = genfromtxt('data_banknote_authentication.csv', delimiter=',', skip_header=1)
# X, y = data[:, :2], data[:, -1]

dataset = pd.read_csv("data_banknote_authentication.csv")
print(dataset.head(5))
x, y = dataset.drop('class', axis=1), dataset['class']

# Separate input data into two classes based on labels
# class_0 = np.array(X[y==0])
# class_1 = np.array(X[y==1])

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=5)

mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
                    solver='sgd', verbose=10, random_state=1,
                    learning_rate_init=.1)

mlp.fit(X_train, y_train)

print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))
fig, axes = plt.subplots(4, 4)

# use global min / max to ensure all weights are shown on the same scale
vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()

for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 5), cmap=plt.cm.gray, vmin=.5 * vmin, vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())

print("DUPSĄ")
plt.show()