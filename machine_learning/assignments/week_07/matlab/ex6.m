clear;
load('ex6data1');

for C = 1:20:100
  model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
  visualizeBoundaryLinear(X, y, model);
  pause;
end
