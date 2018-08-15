#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>
#include "logistic_regression.hpp"

using namespace Eigen;
using namespace std;


int main()
{
  // int n = 3;    //Number of features
  // int m = 100;   //Number of training examples

  // MatrixXf X(m, n);
  // VectorXf y(m);

  // string line;
  // string filename = "ex2data1.txt";

  int n = 1;    //Number of features
  int m = 97;   //Number of training examples

  MatrixXf X(m, n);
  VectorXf y(m);

  string line;
  string filename = "../../week2/c++/ex1data1.txt";

  load(X, y, filename, m, n+1);

  // for(int i=0; i<m; i++) {
  //   for(int j=0; j<n; j++) {
  //     if(j < n-1)
  //       cout << X(i, j) << " ";
  //     else
  //       cout << y(i) << endl;
  //   }
  // }

  
  // MatrixXf Z(m,n);
  // Z = X.unaryExpr(&sigmoid);
  // cout << Z << endl;

  MatrixXf X_aug(m, n+1);
  designMatrix(X, X_aug);

  cout << X << endl;

  return 0;
}