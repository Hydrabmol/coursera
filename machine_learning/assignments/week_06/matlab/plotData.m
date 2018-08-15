function plotData(X, y)
  figure;
  plot(X, y, 'rx', 'MarkerSize', 5);
  
  xlabel('Change in water level');
  ylabel('Water flowing out of the dam');
end
  