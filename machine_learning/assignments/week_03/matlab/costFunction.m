function [ J, grad ] = costFunction( X, y, theta )

  m = length(y);
  
  
  J = (1/m)*(-y'*log(sigmoid(X*theta))-(1-y)' *log(1-sigmoid(X*theta)));
  
  grad = (1/m) * sum( X .* repmat(sigmoid(X*theta) - y, 1, size(X, 2)));
  
  grad = (1 / m) * sum( X .* repmat((sigmoid(X*theta) - y), 1, size(X,2)) );

  
end

