#include <Eigen/Core>
#include <iostream>

namespace ei = Eigen;

// [circulant_func]
template<class ArgType>
class circulant_functor {
  const ArgType &m_vec;
public:
  circulant_functor(const ArgType& arg) : m_vec(arg) {}

  const typename ArgType::Scalar& operator() (ei::Index row, ei::Index col) const {
    ei::Index index = row - col;
    if (index < 0) index += m_vec.size();
    return m_vec(index);
  }
};
// [circulant_func]

// [square]
template<class ArgType>
struct circulant_helper {
  typedef ei::Matrix<typename ArgType::Scalar,
                 ArgType::SizeAtCompileTime,
                 ArgType::SizeAtCompileTime,
                 ei::ColMajor,
                 ArgType::MaxSizeAtCompileTime,
                 ArgType::MaxSizeAtCompileTime> MatrixType;
};
// [square]

// [makeCirculant]
template <class ArgType>
ei::CwiseNullaryOp<circulant_functor<ArgType>, typename circulant_helper<ArgType>::MatrixType>
makeCirculant(const ei::MatrixBase<ArgType>& arg)
{
  typedef typename circulant_helper<ArgType>::MatrixType MatrixType;
  return MatrixType::NullaryExpr(arg.size(), arg.size(), circulant_functor<ArgType>(arg.derived()));
}
// [makeCirculant]

// [main]
int main()
{
  ei::VectorXd vec(4);
  vec << 1, 2, 4, 8;
  ei::MatrixXd mat;
  mat = makeCirculant(vec);
  std::cout << mat << std::endl;
}
// [main]
