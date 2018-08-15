#include <iostream>
#include <fstream>
#include <sstream>
#include "linear_regression.hpp"

using namespace std;



int main() {

  /*
    This should later be changed to be read from the command line using argv or maybe boost.
    It would be great to pass the file only and having the program to compute m and n.
  */
  int n = 2;    //Number of features
  int m = 97;   //Number of training examples

  array2d<float> X(m,n);


  // Right now what follows will only print what's in the file.
  // To try to store to the array read the conversation at: http://www.cplusplus.com/forum/unices/112048/
  string line;
  ifstream myfile ("ex1data1.txt");
  

  int col = 0;
  int row = 0;
  while( getline( myfile, line ) && row < m )
  {
    istringstream iss( line );
    string result;
    for(int c=0; c<n; c++) {
      getline( iss, result, ',' );
      //cout << result << " ";
      X.at(row, c) = float(atof( result.c_str() ));
    }
    row++;
  }

  
  array2d<float> X_augment(X.getRows(), X.getColumns()+1);
  array2dAugment(X_augment, X);

  array2d<float> avg(X_augment.getColumns());
  mean(avg, X_augment);

    

  
  // cout << endl;for(int i=0; i<m; i++)
  // {
  //   for(int j=0; j<n+1; j++)
  //     cout << X_augment.at(i, j) << " ";
  //   cout << endl;
  // }

  // cout << endl;

  // for(int i=0; i<X_augment.getColumns(); i++)
  // {
  //   cout << avg.at(i) << " ";
  // }


  // ofstream outfile("blob", ios_base::binary);
  // outfile.write((char*)X.ptr(), X.getSize());
  // outfile.close();

  
  array2d<float> A(3);
  array2d<float> B(3);
  array2d<float> C(3);
  C.zeros();


  A.at(0) = 1.0; 
  A.at(1) = 1.0;
  A.at(2) = 1.0;

  B.at(0) = 2.0;
  B.at(1) = 2.0;
  B.at(2) = 2.0;

  sum(A, B, C);

  for(int j=0; j<3; j++)
    cout << C.at(j) << " ";
  cout << endl;

  //cout << C.at(0) << endl;



  // array2d<float> A(2, 2);
  // array2d<float> B(2, 2);
  // array2d<float> C(2, 2);
  // C.zeros();


  // A.at(0, 0) = 1.0; 
  // A.at(0, 1) = 1.0;
  // A.at(1, 0) = 1.0; 
  // A.at(1, 1) = 1.0;
  

  // B.at(0, 0) = 2.0; 
  // B.at(0, 1) = 2.0;
  // B.at(1, 0) = 2.0; 
  // B.at(1, 1) = 2.0;

  // dot(A, B, C);

  // for(int i=0; i<2; i++) {
  //   for(int j=0; j<2; j++)
  //       cout << C.at(i, j) << " ";
  //   cout << endl;
  // }




  return 0;
}