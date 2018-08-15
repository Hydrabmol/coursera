#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "linear_regression.hpp"

using namespace Eigen;
using namespace std;

int main()
{
  int n = 2;    //Number of features
  int m = 47;   //Number of training examples

  MatrixXf X(m, n);
  VectorXf y(m);

  string line;
  string filename = "ex1data2.txt";
  //ifstream myfile ("ex1data1.txt");

  load(X, y, filename, m, n+1);

  
  VectorXf mu;
  VectorXf sigma;


  MatrixXf X_norm;
  featureNormalize(X, X_norm, mu, sigma);

  cout << sigma << endl;
  
  MatrixXf X_aug(m, n+1);
  designMatrix(X_norm, X_aug);

  VectorXf theta = VectorXf::Zero(n+1);
  //cout << computeCost(X_aug, y, theta) << endl;

  float alpha = 0.01;
  int iterations = 1500;
  VectorXf J(iterations);
  gradientDescent(X_aug, y, theta, alpha, iterations, J);

  cout << theta << endl << endl;

  Vector2f d;
  d << 1650, 3;

  d = (d - mu).array() / sigma.array();

  Vector3f d_aug;
  d_aug << 1, d[0], d[1];


  float result = theta.transpose() * d_aug; 
  cout << "Result = " << result << endl;

  return 0;
}