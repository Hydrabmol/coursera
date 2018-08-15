load('ex7faces.mat');

[m, n] = size(X);
[X_norm, mu, sigma] = myFeatureNormalize(X);
U = pca(X_norm);

displayData(X(1:100, :));

Z = projectData(X_norm, U, 100);
X_rec = recoverData(Z, U, 100);

figure;
subplot(1, 2, 1);
displayData(X_norm(1:100, :));
hold on;
subplot(1, 2, 2);
displayData(X_rec(1:100, :));
