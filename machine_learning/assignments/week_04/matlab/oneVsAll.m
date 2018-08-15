function [all_theta] = oneVsAll(X, y, num_labels, lambda) 

  [m, n] = size(X); 
  
  X = [ones(m, 1), X];
  all_theta = zeros(num_labels, n+1);  
  
  initial_theta = zeros(n+1, 1);
  
  options = optimset('GradObj', 'on', 'MaxIter', 50);
  
  for c=1:num_labels
    y_c = y == c;
    all_theta(c, :) = fmincg(@(t)(lrCostFunction(t, X, y_c, lambda)), initial_theta, options);
  end
    