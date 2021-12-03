#include <iostream>
#include <Eigen/SVD>

using Eigen::Matrix4f;

float inv_cond(const Eigen::Ref<const Eigen::MatrixXf>& a)
{
  const Eigen::VectorXf sing_vals = a.jacobiSvd().singularValues();
  return sing_vals(sing_vals.size()-1) / sing_vals(0);
}

int main()
{
  Matrix4f m = Matrix4f::Random();
  std::cout << "matrix m:" << std::endl << m << std::endl << std::endl;
  std::cout << "inv_cond(m):          " << inv_cond(m)                      << std::endl;
  std::cout << "inv_cond(m(1:3,1:3)): " << inv_cond(m.topLeftCorner(3,3))   << std::endl;
  std::cout << "inv_cond(m+I):        " << inv_cond(m+Matrix4f::Identity()) << std::endl;
}
