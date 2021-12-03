#include <Eigen/Dense>
#include <iostream>

using Eigen::MatrixXf;
using std::endl;

int main()
{
  MatrixXf m(2,2);
  MatrixXf n(2,2);
  MatrixXf result(2,2);

  m << 1,2,
       3,4;
  n << 5,6,
       7,8;

  result = m * n;
  std::cout << "-- Matrix m*n: --" << endl << result << endl << endl;
  result = m.array() * n.array();
  std::cout << "-- Array m*n: --" << endl << result << endl << endl;
  result = m.cwiseProduct(n);
  std::cout << "-- With cwiseProduct: --" << endl << result << endl << endl;
  result = m.array() + 4;
  std::cout << "-- Array m + 4: --" << endl << result << endl << endl;
}
