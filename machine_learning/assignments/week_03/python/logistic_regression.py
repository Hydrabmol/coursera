import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(np.negative(x)))


def costFunction(X, y, theta):
	m = len(y)
	return (np.dot(np.negative(y), np.log(sigmoid(np.dot(X, theta)))) - np.dot((1- y), np.log(1 - sigmoid(np.dot(X, theta))))) / m

# def gradientDescent(X, y, theta, alpha, iterations):
# 	m = len(y)
# 	J = np.zeros([iterations, 1])

# 	for i in range(0, iterations):
# 		delta = np.sum(np.tile(sigmoid(np.dot(X, theta)) - y.reshape([m, 1]), X.shape[1]) * X, axis=0) / m
# 		theta = theta - alpha * delta.reshape([delta.shape[0], 1])
# 		J[i] = costFunction(X, y, theta)

# 	return [theta, J]


def gradientDescent(X, y, theta, alpha, iterations):
	m = len(y)
	J = np.zeros([iterations, 1])
	for i in range(iterations):
		#print i
		delta = np.sum(np.tile(sigmoid(np.dot(X, theta))-y.reshape([m, 1]), X.shape[1]) * X, axis=0) / m
		theta = theta - alpha * delta.reshape([delta.shape[0], 1])
		#print theta
		J[i] = costFunction(X, y, theta)
	return [theta, J]

data = np.loadtxt('ex2data1.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
m = len(y)

X = np.column_stack((np.ones([m, 1]), X))

theta = np.zeros([3, 1])

print costFunction(X, y, theta)

#alpha = 0.01
#iterations = 400
alpha = 0.003
iterations = 600000

#iterations = 1500
[theta, J] = gradientDescent(X, y, theta, alpha, iterations)
print theta
print J[iterations-1]	 