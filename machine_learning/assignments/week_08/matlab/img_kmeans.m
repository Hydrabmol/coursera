img = double(imread('bird_small.png'));

img = img / 255;

[m, n, c] = size(img);

X = reshape(img, m * n, 3);

K = 16;

initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, 10);
idx = findClosestCentroids(X, centroids);
X_recovered = centroids(idx, :);
X_recovered = reshape(X_recovered, m, n, 3);

subplot(1, 2, 1);
imshow(img);

subplot(1, 2, 2);
imshow(X_recovered);