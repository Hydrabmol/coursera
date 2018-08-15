function [X_norm, mu, sigma] = myFeatureNormalize(X)

    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;

end