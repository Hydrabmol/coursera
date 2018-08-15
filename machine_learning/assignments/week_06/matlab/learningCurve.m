function [ error_train, error_val ] = learningCurve( X, y, Xval, yval, lambda )

  m = size(X, 1);
  
  error_train = zeros(m, 1);
  error_val = zeros(m, 1);
  
  Xval_aug = [ones(size(Xval, 1), 1), Xval];
  
  for i=1:m
    X_aug = [ones(i, 1), X(1:i, :)];  
    theta = trainLinearReg(X_aug, y(1:i), lambda);
    
    [J, grad] = linearRegCostFunction(X_aug, y(1:i), theta, 0);
    error_train(i, 1) = J;
    
    [J, grad] = linearRegCostFunction(Xval_aug, yval, theta, 0);
    error_val(i, 1) = J;
    
  end


end

