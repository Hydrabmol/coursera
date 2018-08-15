data = load('ex2data1.txt');
[m, n] = size(data);
X = [ones(m, 1), data(:, 1:2)];
y = data(:, 3);
initial_theta = zeros(n, 1);

options = optimset('GradObj', 'on', 'MaxIter', 400);
[theta, cost] = fminunc(@(t)(costFunction(X, y, t)), initial_theta, options)

plotData(X(:,2:3), y);
