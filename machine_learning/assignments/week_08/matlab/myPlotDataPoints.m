function myPlotDataPoints(X, idx, K)
  
  %create a palette
  palette = hsv(K + 1);
  colors = palette(idx, 1);
  
  %plot points with palette
  scatter(X(:, 1), X(:, 2), 15, colors);
  
end