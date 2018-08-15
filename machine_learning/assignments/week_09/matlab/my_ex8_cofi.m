% Load data
load('ex8_movies.mat');


% Print average rating for the first film (Toy Story)
fprintf("The average for the first film (Toy Story) is: ");
fprintf("%f\n", mean(Y(1, R(1,:))));


% Collaborative Filtering
load('ex8_movieParams.mat');
%  Reduce the data set size so that this runs faster
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);
lambda = 0;
[J, grad] = cofiCostFunc([X(:), Theta(:)], Y, R, num_users, num_movies, ...
                                  num_features, lambda)

fprintf(['Cost with the loaded parameters: %f\n' ...
        'This should be around:  22.22\n'], J);