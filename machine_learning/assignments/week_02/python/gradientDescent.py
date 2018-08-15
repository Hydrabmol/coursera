import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('ex1data1.txt', dtype=np.float32, delimiter=',')
X = data[:, 0]
y = data[:, 1]
m = len(y)

X = np.column_stack((np.ones([m, 1]), X))

def computeCost(X, y, theta):
	m = len(y)
	return np.sum((np.dot(X, theta)-y.reshape(m, 1))**2) / (2*m)


def gradientDescent(X, y, theta, alpha, iterations):
	m = len(y)
	J = np.zeros([iterations, 1])


	for i in range(iterations):
		delta = np.sum(np.tile(np.dot(X, theta)-y.reshape([m, 1]), X.shape[1]) * X, axis=0) / m
		theta = theta - alpha * delta.reshape([delta.shape[0], 1])
		J[i] = computeCost(X, y, theta)

	return [theta, J]

theta = np.zeros([2, 1])
alpha = 0.01
iterations = 1500
[theta, J] = gradientDescent(X, y, theta, alpha, iterations)
plt.plot(range(iterations), J)
plt.show()