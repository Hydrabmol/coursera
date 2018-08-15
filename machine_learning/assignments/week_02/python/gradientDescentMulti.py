import numpy as np


def featureNormalize(X):
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	return [((X-mu)/sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)]


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


data = np.loadtxt('ex1data2.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
X = X.astype(np.float32)
y = data[:, 2]
y = y.astype(np.float32)
m = len(y)

[X, mu, sigma] = featureNormalize(X)
X = np.column_stack((np.ones([m, 1]), X))


theta = np.zeros([3, 1])
alpha = 0.01
iterations = 400
[theta, J] = gradientDescent(X, y, theta, alpha, iterations)

d = [1650, 3]
d = (d-mu)/sigma
d = d.reshape([d.shape, 1])
d = [1, d]
price = np.dot(d, theta)