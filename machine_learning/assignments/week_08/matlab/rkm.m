function c_i = rkm(X, K, iterations)

    %{
    [m, n] = size(X);


    figure;
    plot(X(:,1), X(:, 2), 'o', 'MarkerSize', 5);

    rows = randi([1, m], K, 1);
    centroids = X(rows, :);
    idx = zeros(m, 1);
    
    fprintf('Program paused. Press enter to continue.\n');
    pause;

    
    
    for i = 1 : iterations
        idx = findClosestCentroids(X, centroids);
        centroids = computeCentroids(X, idx, K);
        hold on;
        plot(X(idx==1, 1), X(idx==1, 2), 'o', 'MarkerSize', 5, 'Color', 'r');
        plot(X(idx==2, 1), X(idx==2, 2), 'o', 'MarkerSize', 5, 'Color', 'g');
        plot(X(idx==3, 1), X(idx==3, 2), 'o', 'MarkerSize', 5, 'Color', 'k');
        plot(centroids(1, 1), centroids(1, 2), 'x', 'MarkerSize', 10, 'Color', 'r');
        plot(centroids(2, 1), centroids(2, 2), 'x', 'MarkerSize', 10, 'Color', 'g');
        plot(centroids(3, 1), centroids(3, 2), 'x', 'MarkerSize', 10, 'Color', 'k');
        hold off;

        fprintf('Program paused. Press enter to continue.\n');
        pause;
    end
   %} 
   
   [m, n] = size(X);
   
   % rows = randi([1, m], K, 1);
   % centroids = X(rows, :);
   centroids = kMeansInitCentroids(X, K);
   idx = zeros(m, 1);
   
   figure;
   hold on;
   
   for i = 1 : iterations
      previous_centroids = centroids;
      idx = findClosestCentroids(X, centroids);
      centroids = computeCentroids(X, idx, K);
      
      myPlotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
      
      fprintf('Program paused. Press enter to continue.\n');
      pause;
      
   end
   

end