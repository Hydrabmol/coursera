#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "linear_regression.hpp"

using namespace Eigen;
using namespace std;

int main()
{
  int n = 1;    //Number of features
  int m = 97;   //Number of training examples

  MatrixXf X(m, n);
  VectorXf y(m);

  string line;
  string filename = "ex1data1.txt";
  //ifstream myfile ("ex1data1.txt");

  load(X, y, filename, m, n+1);

  // for(int i=0; i<m; i++) {
  //   for(int j=0; j<n; j++) {
  //     if(j < n-1)
  //       cout << X(i, j) << " ";
  //     else
  //       cout << y(i) << endl;
  //   }

  // }


  VectorXf theta = VectorXf::Zero(n+1);

  MatrixXf X_aug(m, n+1);

  designMatrix(X, X_aug);

  cout << computeCost(X_aug, y, theta) << endl;

  float alpha = 0.01;
  int iterations = 1500;
  VectorXf J(iterations);
  gradientDescent(X_aug, y, theta, alpha, iterations, J);

  cout << theta << endl;

  VectorXf x1(2), x2(2);
  x1 << 1.0, 3.5; 
  x2 << 1.0, 7;

  cout << theta.transpose() * x1 << endl;
  cout << theta.transpose() * x2 << endl;

  

  return 0;
}