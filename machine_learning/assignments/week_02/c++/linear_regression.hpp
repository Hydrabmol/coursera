#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;



void load(MatrixXf& X, VectorXf& y, string filename, int m, int n)
{
  ifstream myfile(filename.c_str());

  string line;	
  int col = 0;
  int row = 0;
  while( getline( myfile, line ) && row < m )
  {
    istringstream iss( line );
    string result;
    for(int c=0; c<n; c++) {
      getline( iss, result, ',' );
      if(c < n-1)
      	X(row, c) = float(atof( result.c_str() ));
      else
      	y(row) = float(atof( result.c_str() ));
    }
    row++;
  }

}


void designMatrix(MatrixXf& X, MatrixXf& X_aug)
{
  for(int i=0; i<X_aug.rows(); i++)
    X_aug(i, 0) = 1.0;

  for(int i=0; i<X.rows(); i++)
    for(int j=0; j<X.cols(); j++)
      X_aug(i, j+1) = X(i, j);
}


float computeCost(MatrixXf X, VectorXf y, VectorXf theta)
{
  int m = y.rows();

  MatrixXf temp = X * theta - y;
  float J = (temp.array() * temp.array()).sum();
  
  return J / (2*m);
}



void gradientDescent(MatrixXf X, VectorXf y, VectorXf& theta, float alpha, int iterations, VectorXf& J)
{
  int m = y.rows();

  for(int i=0; i<iterations; i++) {
    VectorXf delta = (X.array() * (X * theta - y).replicate(1, X.cols()).array()).colwise().sum() / m;
    theta = theta - alpha * delta;
    J(i) = computeCost(X, y, theta);
  }
}


void standardDeviation(MatrixXf X, VectorXf& sigma)
{
  VectorXf mean = X.colwise().mean();

  MatrixXf temp = mean.transpose().replicate(X.rows(), 1);

  sigma = ((X - temp).colwise().squaredNorm() / X.rows()).array().sqrt();

}


void featureNormalize(MatrixXf X, MatrixXf& X_norm, VectorXf& mu, VectorXf& sigma)
{

  mu = X.colwise().mean();

  standardDeviation(X, sigma);

  X_norm = (X - mu.transpose().replicate(X.rows(), 1)).array() / sigma.transpose().replicate(X.rows(), 1).array();

}


void normalEqn(MatrixXf X, VectorXf y, VectorXf& theta)
{
  theta = (X.transpose() * X).inverse() * X.transpose() * y;
}