import numpy as np


def sigmoid(z):
  return 1 / (1 + np.exp(np.negative(z)))

def mapFeature(X1, X2):
	degree = 6
	out = np.ones(( len(X1), sum(range(degree+2))))
	curr_column = 1
	for i in range(1, degree+1):
		for j in range(i+1):
			out[:, curr_column] = np.power(X1, i-j) * np.power(X2, j)
			curr_column += 1
	return out

def costFunctionReg(theta, X, y, lmb):
	m = len(y)
	J = 0
	grad  = np.zeros([theta.size, 1])
	# grad = np.sum(np.multiply(X, np.tile(sigmoid(np.dot(X,theta))-y, theta.shape[0])), 0).reshape([X.shape[1], 1]) / m
	# grad[1:len(grad), :] = grad[1:len(grad), :] + (lmb/m) * theta[1:len(theta), :] 
	return ((np.dot(np.negative(y.T), np.log(sigmoid(np.dot(X, theta)))) - np.dot((1- y).T, np.log(1 - sigmoid(np.dot(X, theta))))) / m) + ((lmb/(2*m)) * np.dot(theta[1:len(theta)].T,theta[1:len(theta)]))

	
def featureNormalize(X):
	mu = np.mean(X, axis=0)
	sigma = np.std(X, axis=0)
	return [((X-mu)/sigma).astype(np.float32), mu.astype(np.float32), sigma.astype(np.float32)]


def gradientDescent(X, y, theta, alpha, iterations, lmb):
	m = len(y)
	J = np.zeros([iterations, 1])
	for i in range(iterations):
		grad = np.sum(np.multiply(X, np.tile(sigmoid(np.dot(X,theta))-y, theta.shape[0])), 0).reshape([X.shape[1], 1]) / m
		grad[1:len(grad), :] = grad[1:len(grad), :] + (lmb/m) * theta[1:len(theta), :] 
		#delta = np.sum(np.tile(np.dot(X, theta)-y.reshape([m, 1]), X.shape[1]) * X, axis=0) / m
		theta = theta - alpha * grad
		J[i] = costFunctionReg(theta, X, y, lmb)
	return [theta, J]

	




data = np.loadtxt('ex2data2.txt', dtype=np.float32, delimiter=',')
X = data[:, 0:2]
y = data[:, 2].reshape([len(X), 1])

out = mapFeature(X[:, 0], X[:, 1])

X = out

#X = np.column_stack((np.ones([out.shape[0], 1]), out))

theta = np.zeros([out.shape[1], 1])

J = costFunctionReg(theta, X, y, 0.5)

