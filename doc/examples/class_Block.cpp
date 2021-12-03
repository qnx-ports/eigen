#include <Eigen/Core>
#include <iostream>

namespace ei = Eigen;

template<typename Derived>
ei::Block<Derived>
topLeftCorner(ei::MatrixBase<Derived>& m, int rows, int cols)
{
  return ei::Block<Derived>(m.derived(), 0, 0, rows, cols);
}

template<typename Derived>
const ei::Block<const Derived>
topLeftCorner(const ei::MatrixBase<Derived>& m, int rows, int cols)
{
  return ei::Block<const Derived>(m.derived(), 0, 0, rows, cols);
}

int main(int, char**)
{
  ei::Matrix4d m = ei::Matrix4d::Identity();
  std::cout << topLeftCorner(4*m, 2, 3) << std::endl; // calls the const version
  topLeftCorner(m, 2, 3) *= 5;              // calls the non-const version
  std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
  return 0;
}
