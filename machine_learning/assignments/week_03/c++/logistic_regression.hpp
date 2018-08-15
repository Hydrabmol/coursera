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


float sigmoid(float val) {
  return 1 / (1 + exp(-val));
}