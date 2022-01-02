// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) ???? ????????????????????????????????
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/NNLS>
#include <iostream>

/// Check that 'x' solves the NNLS optimization problem `min ||A*x-b|| s.t. 0 <= x`.
/// The \p tolerance parameter is the absolute tolerance on the gradient, A'*(A*x-b).
template <typename MatrixType, typename VectorB, typename VectorX, typename Scalar>
void verify_nnls_optimality(const MatrixType &A, const VectorB &b, const VectorX &x, const Scalar tolerance) {
  // The NNLS optimality conditions are:
  //
  // * 0 = A'*A*x - A'*b - lambda
  // * 0 <= x[i] \forall i
  // * 0 <= lambda[i] \forall i
  // * 0 = x[i]*lambda[i] \forall i
  //
  // we don't know lambda, but by assuming the first optimality condition is true,
  // we can derive it and then check the others conditions.
  const VectorX lambda = A.transpose() * (A * x - b);

  // NNLS solutions are EXACTLY not negative.
  VERIFY_LE(0, x.minCoeff());

  // Exact lambda would be non-negative, but computed lambda might leak a little
  VERIFY_LE(-tolerance, lambda.minCoeff());

  // x[i]*lambda[i] == 0 <~~> (x[i]==0) || (lambda[i] is small)
  VERIFY(((x.array() == Scalar(0)) || (lambda.array() <= tolerance)).all());
}

template <typename MatrixType, typename VectorB, typename VectorX>
void test_nnls_known_solution(const MatrixType &A, const VectorB &b, const VectorX &x_expected) {
  using Scalar = typename MatrixType::Scalar;

  using std::sqrt;
  const Scalar tolerance = sqrt(Eigen::GenericNumTraits<Scalar>::epsilon());
  Index max_iter = 5 * A.cols();  // A heuristic guess.
  NNLS<MatrixType> nnls(A, max_iter, tolerance);
  const bool solved = nnls.solve(b);
  const auto x = nnls.x();

  VERIFY(solved);
  VERIFY_IS_APPROX(x, x_expected);
  verify_nnls_optimality(A, b, x, tolerance);
}

template <typename MatrixType>
void test_nnls_random_problem() {
  //
  // SETUP
  //

  Index cols = MatrixType::ColsAtCompileTime;
  if (cols == Dynamic) cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  Index rows = MatrixType::RowsAtCompileTime;
  if (rows == Dynamic) rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);
  VERIFY_LE(cols, rows);  // To have a unique LS solution: cols <= rows.

  // Make some sort of random test problem from a wide range of scales and condition numbers.
  using std::pow;
  using Scalar = typename MatrixType::Scalar;
  const Scalar sqrtConditionNumber = pow(Scalar(10), internal::random<Scalar>(Scalar(0), Scalar(2)));
  const Scalar scaleA = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const Scalar minSingularValue = scaleA / sqrtConditionNumber;
  const Scalar maxSingularValue = scaleA * sqrtConditionNumber;
  const auto svs = setupRangeSvs<Matrix<Scalar, Dynamic, 1>>(cols, minSingularValue, maxSingularValue);
  MatrixType A(rows, cols);
  generateRandomMatrixSvs(svs, rows, cols, A);

  // Make a random RHS also with a random scaling.
  using VectorB = decltype(A.col(0).eval());
  const Scalar scaleB = pow(Scalar(10), internal::random<Scalar>(Scalar(-3), Scalar(3)));
  const VectorB b = scaleB * VectorB::Random(A.rows());

  //
  // ACT
  //

  using Scalar = typename MatrixType::Scalar;
  using std::sqrt;
  const Scalar tolerance =
      sqrt(Eigen::GenericNumTraits<Scalar>::epsilon()) * b.cwiseAbs().maxCoeff() * A.cwiseAbs().maxCoeff();
  Index max_iter = 5 * A.cols();  // A heuristic guess.
  NNLS<MatrixType> nnls(A, max_iter, tolerance);
  const bool solved = nnls.solve(b);
  const auto x = nnls.x();

  //
  // VERIFY
  //

  VERIFY(solved);  // In fact, NNLS can fail on some problems, but they are rare in practice.
  verify_nnls_optimality(A, b, x, tolerance);
}

// 4x2 problem, unconstrained solution positive
void test_nnls_known_1() {
  Matrix<double, 4, 2> A(4, 2);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 2, 1> x(2);
  A << 1, 1, 2, 4, 3, 9, 4, 16;
  b << 0.6, 2.2, 4.8, 8.4;
  x << 0.1, 0.5;

  return test_nnls_known_solution(A, b, x);
}

// 4x3 problem, unconstrained solution positive
void test_nnls_known_2() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13;

  test_nnls_known_solution(A, b, x);
}

// Simple 4x4 problem, unconstrained solution non-negative
void test_nnls_known_3() {
  Matrix<double, 4, 4> A(4, 4);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 4, 1> x(4);

  A << 1, 1, 1, 1, 2, 4, 8, 16, 3, 9, 27, 81, 4, 16, 64, 256;
  b << 0.73, 3.24, 8.31, 16.72;
  x << 0.1, 0.5, 0.13, 0;

  test_nnls_known_solution(A, b, x);
}

// Simple 4x3 problem, unconstrained solution non-negative
void test_nnls_known_4() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.23, 1.24, 3.81, 8.72;
  x << 0.1, 0, 0.13;

  test_nnls_known_solution(A, b, x);
}

// Simple 4x3 problem, unconstrained solution indefinite
void test_nnls_known_5() {
  Matrix<double, 4, 3> A(4, 3);
  Matrix<double, 4, 1> b(4);
  Matrix<double, 3, 1> x(3);

  A << 1, 1, 1, 2, 4, 8, 3, 9, 27, 4, 16, 64;
  b << 0.13, 0.84, 2.91, 7.12;
  // Solution obtained by original nnls() implementation in Fortran
  x << 0.0, 0.0, 0.1106544;

  test_nnls_known_solution(A, b, x);
}

void test_known_problems() {
  test_nnls_known_1();
  test_nnls_known_2();
  test_nnls_known_3();
  test_nnls_known_4();
  test_nnls_known_5();
}

void test_nnls_with_half_precision() {
  // The random matrix generation tools don't work with `half`,
  // so here's a simpler setup mostly just to check that NNLS compiles & runs with custom scalar types.

  using Mat = Matrix<half, 4, 4>;
  using Vec = Matrix<half, 4, 1>;
  Mat A = Mat::Random();
  Vec b = Vec::Random();

  NNLS<Mat> nnls(A, 20, half(1e-2f));
  const bool solved = nnls.solve(b);
  const auto x = nnls.x();

  VERIFY(solved);
  verify_nnls_optimality(A, b, x, half(1e-1));
}

EIGEN_DECLARE_TEST(NNLS) {
  test_known_problems();
  test_nnls_with_half_precision();
  for (int i = 0; i < g_repeat; i++) {
    test_nnls_random_problem<MatrixXf>();
    test_nnls_random_problem<MatrixXd>();
    using MatFixed = Matrix<double, 12, 5>;
    test_nnls_random_problem<MatFixed>();
  }
}
