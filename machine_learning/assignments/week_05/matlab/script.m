input_size = 400;
hidden_layer_size = 25;
num_labels = 10;

load('ex4data1.mat');


initial_theta1 = randInitializeWeights(0.12, input_layer_size, hidden_layer_size);
initial_theta2 = randInitializeWeights(0.12, hidden_layer_size, num_labels); 

initial_nn_params = [initial_theta1(:); initial_theta2(:)];

options = optimset('MaxIter', 400);
lambda = 1;

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
                                   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

Theta1 = reshape(nn_params(1:(hidden_layer_size*(input_size+1))), hidden_layer_size, input_size+1);
Theta2 = reshape(nn_params((hidden_layer_size*(input_size+1)+1) : end), num_labels, hidden_layer_size +1 )

res = pred(Theta1, Theta2, X);

m = size(X, 1);

for i=1:m
    yVec(y(i), i) = 1;
end

fprintf('\nTraining Set Accuracy: %f\n', mean(double(res == yVec)) * 100);


