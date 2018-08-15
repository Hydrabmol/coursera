#include <iostream>

using namespace std;

template<typename T>
class array2d
{
  public:
  array2d(const unsigned h,const unsigned w) :  rows(h) , columns(w) {
    buffer = new T[rows * columns];
    size = rows*columns*sizeof(T);
  }
  array2d(const unsigned int bytes) {
    buffer = new T[bytes];
    size = bytes;
    rows = 1;
    columns = bytes;
  }
  virtual ~array2d() {
    delete[] buffer;
  }
  T& at(const unsigned i,const unsigned j) {
    unsigned idx = i * columns + j;
    return buffer[idx];
  }
  T& at(const unsigned i) {
    return buffer[i];
  }
  int getRows() const {
    return rows;
  }
  int getColumns() const {
    return columns;
  }
  int getSize() const {
    return size;
  }
  T* ptr() {
    return buffer;
  }
  void zeros() {
    for(int i=0; i<rows; i++)
      for(int j=0; j<columns; j++)
        buffer[i * columns + j] = 0.0;
  }


private:
  unsigned rows;
  unsigned columns;
  unsigned int size;
  T* buffer;
};


// array2d<float>* array2dAugment(array2d<float>& X) 
// {
//   array2d<float>* X_augment = new array2d<float>(X.getRows(), X.getColumns()+1);

//   for(int i=0; i<X_augment->getRows(); i++)
//     X_augment->at(i, 0) = 1.0;

//   for(int i=0; i<X_augment->getRows(); i++)
//     for(int j=1; j <X_augment->getColumns(); j++)
//       X_augment->at(i, j) = X.at(i, j-1);

//   return X_augment;  
// }


void array2dAugment(array2d<float>& X_augment, array2d<float>& X) 
{
  for(int i=0; i<X_augment.getRows(); i++)
    X_augment.at(i, 0) = 1.0;

  for(int i=0; i<X_augment.getRows(); i++)
    for(int j=1; j <X_augment.getColumns(); j++)
      X_augment.at(i, j) = X.at(i, j-1);
}



void mean(array2d<float>& avg, array2d<float>& X)
{
  for(int j=0; j<X.getColumns(); j++)
  {
    float count = 0;
    for(int i=0; i<X.getRows(); i++)
    {
      count += X.at(i, j);

    }
    avg.at(j) = count/X.getRows();
  }  
}


void dot(array2d<float>& A, array2d<float>& B, array2d<float>& C) 
{
  if(A.getRows() == 1 && B.getRows() == 1 && A.getColumns() == B.getColumns())
  {
    for(int j=0; j<A.getColumns(); j++) 
      C.at(0) += A.at(j) * B.at(j);

    return;  
  }
  
  if(A.getColumns() != B.getRows())
  {
    cout << "Error! Number of rows and columns mismatch." << endl;
    exit(EXIT_FAILURE);
  }

  for(int i=0; i<A.getRows(); i++) 
    for(int k=0; k<B.getColumns(); k++) 
      for(int j=0; j<A.getColumns(); j++) 
          C.at(i, k) += A.at(i, j) * B.at(j, k);

}


void sum(array2d<float>& A, array2d<float>& B, array2d<float>& C)
{
  if(A.getRows() != B.getRows() || A.getRows() != C.getRows() || A.getColumns() != B.getColumns() || A.getColumns() != C.getColumns())
  {
    cout << "Error! Dimensions mismatch." << endl;
    exit(EXIT_FAILURE);
  }

  for(int i=0; i<C.getRows(); i++)
    for(int j=0; j<C.getColumns(); j++)
      if(C.getRows() != 1)
        C.at(i, j) = A.at(i, j) + B.at(i, j);
      else
        C.at(j) = A.at(j) + B.at(j);
}






