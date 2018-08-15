import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.optimize import minimize


def sigmoid(z):
	return 1 / (1 + np.exp(-z))


def cost(theta, X, y, lmb):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	m = X.shape[0]
	return (-y.T * np.log(sigmoid(X * theta.T)) - (1 - y).T * np.log(1 - sigmoid(X * theta.T)))  / m + lmb * np.sum(np.power( theta[1:len(theta.T), 0], 2)) / (2*m)



def gradient(theta, X, y, lmb):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	m = X.shape[0]
	error = sigmoid(X * theta.T) - y
	grad = (X.T * error) / m
	grad[1:grad.shape[0], 0] = grad[1:grad.shape[0], 0] + lmb * np.sum(np.power(theta[1:theta.shape[0], 0], 2)) / (2*m) 
	return grad


def one_vs_all(X, y, num_labels, lmb):
	rows  = X.shape[0]
	params = X.shape[1]


	all_theta = np.zeros([num_labels, params + 1])


	X = np.insert(X, 0, values=np.ones(rows), axis=1)

	theta = np.zeros(params + 1)
	y_i = np.array([1 if label == 0 else 0 for label in y])
	y_i = np.reshape(y_i, (rows, 1))


	for i in range(1, num_labels+1):
		theta = np.zeros(params + 1)
		y_i = np.array([1 if label == i else 0 for label in y])
		y_i = np.reshape(y_i, (rows, 1))

		fmin = minimize(fun=cost, x0=theta, args=(X, y_i, lmb), method='TNC', jac=gradient)
		all_theta[i-1,:] = fmin.x

	return all_theta


def predict_all(X, all_theta):
	rows = X.shape[0]
	params = X.shape[1]
	num_labels = all_theta.shape[0]

	X = np.insert(X, 0, values=np.ones(rows), axis=1)

	X = np.matrix(X)
	all_theta = np.matrix(all_theta)

	h = sigmoid(X * all_theta.T)

	h_argmax = np.argmax(h, axis=1)

	h_argmax = h_argmax + 1

	return h_argmax

data = loadmat('ex3data1.mat')

all_theta = one_vs_all(data['X'], data['y'], 10, 1)

y_pred = predict_all(data['X'], all_theta)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, data['y'])]

accuracy = (sum(map(int, correct))) / float(len(correct))

print 'accuracy = {0}%'.format(accuracy * 100)


