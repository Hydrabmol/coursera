function grad = sigmoidGradient(z)
  
  grad = sigmoid(z) .* (1 - sigmoid(z));