import numpy as np


def normalEquation(X, y):
	return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)), y)


data = np.loadtxt('ex1data2.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)
y = y.reshape((m, 1))
X = np.column_stack((np.ones([m, 1]), X))

theta = normalEquation(X, y)
