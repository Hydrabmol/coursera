function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% function J = nnCostFunction(num_labels, X, y, Theta1, Theta2)

  [m, n] = size(X);
  yVec = zeros(num_labels, m);

  Theta1 = reshape(nn_params(1 : hidden_layer_size*(input_layer_size + 1)), hidden_layer_size, input_layer_size + 1);
  Theta2 = reshape(nn_params((hidden_layer_size * (input_layer_size+1) + 1 ): end), num_labels, hidden_layer_size+1);  
  
  for i=1:m
    yVec(y(i), i) = 1;
  end
  
  a1 = [ones(m, 1) X];
  z2 = Theta1 * a1';
  a2 = sigmoid(z2)';
  a2 = [ones(size(a2, 1), 1), a2]';
  z3 = Theta2 * a2;
  a3 = sigmoid(z3);

  
  
  J = 0;
  for i=1:m
    J = J - yVec(:, i)' *log(a3(:, i)) - (1 - yVec(:, i))' * log(1 - a3(:, i));
  end
  
  J = J / m;
  %J = J + lambda * ( sum(size(Theta1(:, 2:end))) +  sum(size(Theta2(:, 2:end)))) / m;
  J = J + lambda * (sum(sum(Theta1(:, 2:end) .^ 2)) +  sum(sum( Theta2(:, 2:end) .^ 2))) / (2*m);
  
  
  
  Theta1_grad = zeros(size(Theta1));
  Theta2_grad = zeros(size(Theta2));
  
  for t=1:m
    a1 = [1; X(t, :)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    delta_3 = a3 - yVec(:, t);
    
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)]; 
    delta_2 = delta_2(2:end);
   
    Theta1_grad = Theta1_grad + delta_2 * a1';
	  Theta2_grad = Theta2_grad + delta_3 * a2'; 
  end
  
  Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
  Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
  
  grad = [Theta1_grad(:); Theta2_grad(:)];
  
    
end
  