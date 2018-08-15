data = load('ex2data2.txt')
X = data(:, 1:2)
y = data(:, 3)

out = mapFeature(X(:,1), X(:, 2))