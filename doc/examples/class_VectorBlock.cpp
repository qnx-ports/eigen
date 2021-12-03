#include <Eigen/Core>
#include <iostream>

namespace ei = Eigen;

template<typename Derived>
ei::VectorBlock<Derived>
segmentFromRange(ei::MatrixBase<Derived>& v, int start, int end)
{
  return ei::VectorBlock<Derived>(v.derived(), start, end-start);
}

template<typename Derived>
const ei::VectorBlock<const Derived>
segmentFromRange(const ei::MatrixBase<Derived>& v, int start, int end)
{
  return ei::VectorBlock<const Derived>(v.derived(), start, end-start);
}

int main(int, char**)
{
  ei::Matrix<int,1,6> v; v << 1,2,3,4,5,6;
  std::cout << segmentFromRange(2*v, 2, 4) << std::endl; // calls the const version
  segmentFromRange(v, 1, 3) *= 5;              // calls the non-const version
  std::cout << "Now the vector v is:" << std::endl << v << std::endl;
  return 0;
}
