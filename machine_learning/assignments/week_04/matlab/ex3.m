load('ex3data1);

lambda = 0.5;
num_labels = 10;

all_theta = oneVsAll(X, y, num_labels, lambda);

p = predictOneVsAll(all_theta, X);

accuracy = sum(p - y == 0) / size(X, 1);

load('ex3weights');

p = predict(Theta1, Theta2, X);

accuracy_nn = sum(p - y == 0) / size(X, 1);