function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  
m = size(X, 1);

J = ((X * theta - y)' * (X * theta - y)) / (2*m) + lambda * (theta(2:end, 1)' * theta(2:end, 1)) / (2*m);

grad = (sum(repmat(X * theta - y, 1, size(X, 2)) .* X)/m)';
grad(2:end) = grad(2:end) + lambda * theta(2:end) / m;

 
end