function myPlotProgresskMeans(X, centroids, previous, idx, K, i)

  % plot the examples
  myPlotDataPoints(X, idx, K);
  
  % plot the centroids
  plot(centroids(:, 1), centroids(:, 2), 'x', 'MarkerSize', 10, 'linewidth', 3, 'MarkerEdgeColor', 'k');
 
   
  % plot the centroids history
  for j = 1 : size(centroids)
    drawLine(centroids(j, :), previous(j, :));
  end
  
  % Title
  title(sprintf('Iteration number: %d', i)); 
  

end