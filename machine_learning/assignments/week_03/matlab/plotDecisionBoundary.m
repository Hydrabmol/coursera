function plotDecisionBoundary(theta, X, y)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

  plotData(X(:, 2:3), y);
  hold on;

  if size(X, 2) <= 3
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];
    plot_y = (-1./theta(3)) * ( theta(2)*plot_x + theta(1));
    
    plot(plot_x, plot_y);
    
    legend('Admitted', 'Not Admitted', 'Decision Boundary');
    axis([30, 100, 30, 100]);  
  end
  


end

