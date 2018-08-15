function p = predictOneVsAll(all_theta, X)
  
  [m, n] = size(X);
   
  X = [ones(m, 1), X];
  
  [v, p] = max(sigmoid(all_theta * X'));
  
  p = p';
  
  