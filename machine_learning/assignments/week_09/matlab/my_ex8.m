% Load the dataset
load('ex8data1.mat');

% Plot the examples
plot(X(:,1), X(:,2), 'bx');
axis([0 30 0 30]);
ylabel('Through-put (mb/s)');
xlabel('Latency (ms)');

fprintf('Program paused. Press enter to continue.\n');
pause

% Compute the mean and the variance of our feature
fprintf('Estimating Gaussian Fit');
[mu, sigma2] = estimateGaussian(X)

p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;


% Find outliers
pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon, F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');


% Let's identify the outliers
outliers = find(p < epsilon);
hold on;
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;

% More realistic dataset
clear;
load('ex8data2.mat');
[mu, sigma2] = estimateGaussian(X);
p = multivariateGaussian(X, mu, sigma2);
pval = multivariateGaussian(Xval, mu, sigma2);
[epsilon, F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p < epsilon));
fprintf('   (you should see a value epsilon of about 1.38e-18)\n\n');
pause



