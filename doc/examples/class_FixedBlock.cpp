#include <Eigen/Core>
#include <iostream>

namespace ei = Eigen;

template<typename Derived>
ei::Block<Derived, 2, 2>
topLeft2x2Corner(ei::MatrixBase<Derived>& m)
{
  return ei::Block<Derived, 2, 2>(m.derived(), 0, 0);
}

template<typename Derived>
const ei::Block<const Derived, 2, 2>
topLeft2x2Corner(const ei::MatrixBase<Derived>& m)
{
  return ei::Block<const Derived, 2, 2>(m.derived(), 0, 0);
}

int main(int, char**)
{
  ei::Matrix3d m = ei::Matrix3d::Identity();
  std::cout << topLeft2x2Corner(4*m) << std::endl; // calls the const version
  topLeft2x2Corner(m) *= 2;              // calls the non-const version
  std::cout << "Now the matrix m is:" << std::endl << m << std::endl;
  return 0;
}
