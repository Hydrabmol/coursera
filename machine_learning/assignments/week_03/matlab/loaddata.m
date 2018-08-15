data = load('ex2data1.txt');
[m, n] = size(data);
X = [ones(m, 1), data(:, 1:2)];
y = data(:, 3);

