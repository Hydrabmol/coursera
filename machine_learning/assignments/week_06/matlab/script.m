load('ex5data1.mat');
theta = [1; 1]
lambda = 1

m = size(X, 1);

X_aug = [ones(m, 1), X];

%[J, grad] = linearRegCostFunction(X, y, theta, lambda)

theta = trainLinearReg(X_aug, y, lambda);

predictions = X_aug * theta;


figure;
plot(X, y, 'rx');
hold;
plot(X, predictions, '--');
xlabel('Change in water level');
ylabel('Water flowing out of the dam');

[error_train, error_val] = learningCurve(X, y, Xval, yval, lambda);

figure;
plot([1:m], error_train, 'b');
hold;
plot([1:m], error_val, 'g');
xlabel('Number of training examples');
ylabel('Error');

poly_X = polyFeatures(X, 8);
[poly_X, mu, sigma] = featureNormalize(poly_X);

poly_X_aug = [ones(size(poly_X, 1), 1), poly_X];

theta = trainLinearReg(poly_X_aug, y, 0);

figure
plotFit(min(X), max(X), mu, sigma, theta, 8);
hold on
plot(X, y, 'rx');

poly_Xval = polyFeatures(Xval, 8);
poly_Xval = (poly_Xval - mu) ./ sigma;
poly_Xval_aug = [ones(size(poly_Xval, 1), 1), poly_Xval];

[lambda_vec, error_train, error_val] = validationCurve(poly_X_aug, y, poly_Xval_aug, yval);
figure;
plot(lambda_vec, error_train, 'b');
hold;
plot(lambda_vec, error_val, 'g');

[min_err, lambda_pos] = min(error_val);

theta = trainLinearReg(poly_X_aug, y, lambda_vec(lambda_pos));

poly_Xtest = polyFeatures(Xtest, 8);
poly_Xtest = (poly_Xtest - mu) ./ sigma;
poly_Xtest_aug = [ones(size(poly_Xtest, 1), 1), poly_Xtest];
xlabel('lambda');
ylabel('Error');

error = linearRegCostFunction(poly_Xtest_aug, ytest, theta, 0)

[error_train, error_val] = learningCurveAvg(poly_X, y, poly_Xval, yval, 0.01, 50);

figure;
plot([1:m], error_train, 'b');
hold;
plot([1:m], error_val, 'g');
xlabel('Number of training examples');
ylabel('Error');


