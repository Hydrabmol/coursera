function p = predict(Theta1, Theta2, X)
  
  [m, n] = size(X);
  
  X = [ones(m, 1), X];
  
  z = sigmoid(Theta1 * X');
  
  z = [ones(1, size(z, 2)); z];
 
  [v, p] = max(sigmoid(Theta2 * z));
  
  p = p';