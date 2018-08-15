load('ex4data1.mat');
load('ex4weights.mat');

nn_params = [ Theta1(:); Theta2(:)];

input_layer_size = 400;
hidden_layer_size = 25;
num_labels = 10;

epsilon_init = 0.12;
%Theta1 = rand(input_layer_size, hidden_layer_size + 1) * 2 * epsilon_init - epsilon_init;
%Theta2 = rand(hidden_layer_size, num_labels + 1) * 2 * epsilon_init - epsilon_init;