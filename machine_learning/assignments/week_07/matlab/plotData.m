function plotData(X, y)

  x0 = X(find(y==0), :);
  x1 = X(find(y==1), :);
  
  figure;
  plot(x0(:, 1), x0(:, 2), 'o', 'MarkerFaceColor', 'y', 'MarkerEdgeColor', 'k');
  hold on;
  plot(x1(:, 1), x1(:, 2), '+', 'Color', 'k');
  hold off;

endfunction
