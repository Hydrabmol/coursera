function [ error_train, error_val ] = learningCurveAvg( X, y, Xval, yval, lambda, iters )

  
  m = size(X, 1);
  
  error_train = zeros(m, iters);
  error_val = zeros(m, iters);
  
  
  Xval_aug = [ones(size(Xval, 1), 1), Xval];

  for j=1:iters
      for i=1:m
        X_aug = X(:, randperm(size(X, 2)));  
        X_aug = [ones(size(X_aug, 1), 1), X_aug(1:i, :)];  
        theta = trainLinearReg(X_aug, y(1:i), lambda);

        [J, grad] = linearRegCostFunction(X_aug, y(1:i), theta, 0);
        error_train(i, j) = J;

        [J, grad] = linearRegCostFunction(Xval_aug, yval, theta, 0);
        error_val(i, j) = J;

      end
  end
  
  error_train = sum(error_train) / m;
  error_val = sum(error_val) / m;
  

end

