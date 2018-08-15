import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt



def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return [X_norm, mu, sigma]


def pca(X):
    X = np.matrix(X)
    cov_X = (X.T * X) / X.shape[0]
    [U, S, V] = np.linalg.svd(cov_X)

    return [U, S, V]


def project_data(X, U, K):
    return X * U[:, 0:K]


def recover_data(Z, U, K):
    return Z * U[:, 0:K].T


data = loadmat('ex7data1.mat')
X = data['X']
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(X[:,0], X[:,1])
plt.show()

X_norm = feature_normalize(X)[0]
[U, S, V] = pca(X_norm)
Z = project_data(X_norm, U, 1)
X_recovered = recover_data(Z, U, 1)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter([X_recovered[:, 0]], [X_recovered[:,1]])
plt.show()

# PCA on an image
faces = loadmat('ex7faces.mat')
X = faces['X']
face = np.reshape(X[3,:], (32, 32))
plt.imshow(face, cmap='gray')
plt.show()

X_norm = feature_normalize(X)[0]
[U, S, V] = pca(X_norm)
Z = project_data(X_norm, U, 100)
X_recovered = recover_data(Z, U, 100)
face_recovered = np.reshape(X_recovered[3,:], (32, 32))
plt.imshow(face_recovered, cmap='gray')
plt.show()
