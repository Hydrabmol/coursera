
clear ; close all; clc

data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

[X mu sigma] = featureNormalize(X);
X = [ones(m, 1) X];

alpha = 0.01;
num_iters = 400;

theta = zeros(3, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);


% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

d = [1650 3];
d = (d - mu) ./ sigma;
d = [ones(1, 1) d];
price = d * theta; % You should change this



% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price)

