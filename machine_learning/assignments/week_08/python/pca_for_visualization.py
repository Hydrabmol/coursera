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


def k_means_init_centroids(X, K):
    [m, n] = X.shape
    indices = np.random.randint(m, size=K)
    centroids = X[indices, :]

    return centroids


def find_closest_centroids(X, centroids):
    m = X.shape[0]
    K = centroids.shape[0]
    idx = np.zeros([m, 1])

    for i in range(m):
        distances = np.zeros([K, 1])
        for j in range(K):
            distances[j] = np.sum((X[i, :] - centroids[j, :]) ** 2)
        idx[i, 0] = int(distances.argmin().astype(np.int))

    return idx.astype(np.int)


def compute_centroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros([K, n])

    for i in range(K):
        indices = np.where(idx == i)
        centroids[i] = np.sum(X[indices[0], :], axis=0) / indices[0].shape[0]

    return centroids


def run_k_means(X, initial_centroids, max_iters):
    [m, n] = X.shape
    K = initial_centroids.shape[0]
    idx = np.zeros([m, n])
    centroids = initial_centroids

    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)

    return [idx, centroids]


imagedata = loadmat('bird_small.mat')
A = imagedata['A']
A = A / 255.0
X = A.reshape(A.shape[0]*A.shape[1], A.shape[2])
K = 16
maxiters = 10
initial_centroids = k_means_init_centroids(X, K)
[idx, centroids] = run_k_means(X, initial_centroids, maxiters)

sel = np.random.randint(0, X.shape[0], [1000, 1])
palette = [ 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
color = [palette[idx[i, 0]] for i in range(idx.shape[0])]

X_norm = feature_normalize(X)[0]
[U, S, V] = pca(X_norm)
Z = project_data(X_norm, U, 2)
fig, ax = plt.subplots(figsize=(12,8))
ax.scatter([Z[:, 0]], [Z[:,1]])
plt.show()


print "OK"
