function res = pred(Theta1, Theta2, X)
  
  m = size(X, 1);
  
  a1 = [ones(m, 1) X];
  z2 = Theta1 * a1';
  a2 = sigmoid(z2)';
  a2 = [ones(size(a2, 1), 1), a2]';
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);
  
  res = a3;
  
end