#include <iostream>

using namespace std;


class Array2d
{
    public:

    private:
        float* p;
        int m;
        int n;
};

int main() {

    Array2d* X = loadFeatures(nome_file);
    Array2d* y = loadLabels(nomefile);
    float mu = computeMean(X);
    float sigma = computeStd();
    X_norm = featureNormalize(X, mu, sigma);
    designMatrix(X_norm);

    int iterations = 400;
    float alpha = 0.01;
    Array2d* theta = new array3d(3);
    Array2d* J = new array3d(iterations);
    gradientDescent(X_norm, y, theta, alpha, iterations; J);



    return 0;
}