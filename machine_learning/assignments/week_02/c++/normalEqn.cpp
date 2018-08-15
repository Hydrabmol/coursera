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
  

  load(X, y, filename, m, n+1);

  VectorXf theta = VectorXf::Zero(n+1);

  MatrixXf X_aug(m, n+1);

  designMatrix(X, X_aug);

  //cout << computeCost(X_aug, y, theta) << endl;

  //cout << (X_aug.transpose() * X_aug).inverse() * X_aug.transpose() * y << endl;
   
  normalEqn(X_aug, y, theta);

  cout << theta << endl;

  return 0;
}