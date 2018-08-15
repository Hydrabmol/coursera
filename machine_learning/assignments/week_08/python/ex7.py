import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt



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
            distances[j] = np.sum((X[i, :] - centroids[j, :])**2)
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




data = loadmat('ex7data2.mat')
X = data['X']
K = 3

initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
idx = find_closest_centroids(X, initial_centroids)
print idx[0:3]
print "The values should be: 0, 2, 1"
centroids = compute_centroids(X, idx, K)
print centroids
print('\n(the centroids should be\n');
print('   [ 2.428301 3.157924 ]\n');
print('   [ 5.813503 2.633656 ]\n');
print('   [ 7.119387 3.616684 ]\n\n');

[idx, centroids] = run_k_means(X, initial_centroids, 10)


cluster1 = X[np.where(idx == 0)[0], :]
cluster2 = X[np.where(idx == 1)[0], :]
cluster3 = X[np.where(idx == 2)[0], :]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(cluster1[:, 0], cluster1[:, 1], s=30, color='r', label='Cluster 1')
ax.scatter(cluster2[:, 0], cluster2[:, 1], s=30, color='g', label='Cluster 2')
ax.scatter(cluster3[:, 0], cluster3[:, 1], s=30, color='b', label='Cluster 3')
ax.legend()
plt.show()


# KMEAN on image
imagedata = loadmat('bird_small.mat')
A = imagedata['A']
A = A / 255.0
img_dims = A.shape
X = A.reshape(img_dims[0]*img_dims[1], img_dims[2])
K = 16
initial_centroids = k_means_init_centroids(X, K)
[idx, centroids] = run_k_means(X, initial_centroids, 10)
X_recovered = centroids[idx].reshape(img_dims[0], img_dims[1], img_dims[2])
plt.imshow(X_recovered)
plt.show()