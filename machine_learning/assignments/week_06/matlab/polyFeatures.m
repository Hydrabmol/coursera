function [ poly_X ] = polyFeatures( X, p )

  poly_X = zeros(size(X, 1), p);
  poly_X(:, 1) = X; 
  
  for i=2:p
    poly_X(:, i) = X .^ i;
  end

end

