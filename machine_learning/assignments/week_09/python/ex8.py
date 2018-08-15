import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from math import pi


def estimate_gaussian(X):
    m = X.shape[0]
    mu = np.mean(X, axis=0)
    sigma2 = np.sum(np.power((X - mu), 2), axis=0) / m

    return [mu, sigma2]


def select_threshold(pval, yval):
    best_epsilon = 0.0
    best_F1 = 0.0
    F1 = 0.0

    step = (pval.max() - pval.min()) / 1000
    for epsison in np.arange(pval.min(), pval.max(), step):
        pred = pval < epsison
        pred = pred.reshape(pred.shape[0], 1)

        tp = np.sum(np.logical_and(pred == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(pred == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(pred == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        F1 = (2 * precision * recall) / (precision + recall)

        if F1 > best_F1:
            best_epsilon = epsison
            best_F1 = F1

    return [best_epsilon, best_F1]



def multivariate_gaussian(X, mu, sigma2):
    k = mu.shape[0]

    if 1 in sigma2.shape or len(sigma2.shape) == 1:
        sigma2 = np.diag(sigma2)

    X = X - mu
    p = (2 * pi) ** (- k / 2.0) * np.linalg.det(sigma2) ** (-0.5) * np.exp(-0.5 * np.sum(np.multiply(np.dot(X, np.linalg.pinv(sigma2)), X),1))
    return p






data = loadmat('ex8data1.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

[mu, sigma2] = estimate_gaussian(X)


p = multivariate_gaussian(X, mu, sigma2)

pval = multivariate_gaussian(Xval, mu, sigma2)

[epsilon, F1] = select_threshold(pval, yval)

print epsilon, F1

outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0], 0], X[outliers[0],1], s=50, color='r', marker='o')
plt.show()


data = loadmat('ex8data2.mat')
X = data['X']
Xval = data['Xval']
yval = data['yval']

[mu, sigma2] = estimate_gaussian(X)

p = multivariate_gaussian(X, mu, sigma2)
pval = multivariate_gaussian(Xval, mu, sigma2)

[epsilon, F1] = select_threshold(pval, yval)

print("Epsilon = {}\nF1 score = {}".format(epsilon,F1))

