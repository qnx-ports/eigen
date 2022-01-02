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

template <typename MatrixType, typename VectorB, typename VectorX>
void test_nnls_known_solution(const MatrixType &A, const VectorB &b, const VectorX &x_expected) {
  using Scalar = typename MatrixType::Scalar;

  using std::sqrt;
  const Scalar tolerance = sqrt(Eigen::GenericNumTraits<Scalar>::epsilon());
  Index max_iter = 5 * A.cols();  // A heuristic guess.
  NNLS<MatrixType> nnls(A, static_cast<int>(max_iter), tolerance);
  const bool solved = nnls.solve(b);
  const auto x = nnls.x();

  // Generally, `gradient - Lagrange_multipliers` should be zero.
  // The Lagrange multiplers are unknown, but they are zero when is x not zero.
  // So, gradient[i] should be zero for those `x[i]` that are non-zero.
  auto gradient = (A.transpose() * (A * x - b)).eval();
  Scalar gradient_optimality(0);
  for (Index i = 0; i != x.size(); ++i) {
    if (0 < x[i]) {
      using std::abs;
      using std::max;
      gradient_optimality = (max)(abs(gradient[i]), gradient_optimality);
    }
  }

  VERIFY(solved);
  VERIFY_IS_APPROX(x, x_expected);
  VERIFY_LE(gradient_optimality, tolerance);
  VERIFY_LE(0, x.minCoeff());  // NNLS solutions are not negative.
}

template <typename MatrixType>
void test_nnls_dense_solution() {
  typedef typename NumTraits<typename MatrixType::Scalar>::Real RealScalar;
  Index cols = MatrixType::ColsAtCompileTime;
  if (cols == Dynamic) cols = internal::random<Index>(1, EIGEN_TEST_MAX_SIZE);
  // To have a unique solution: rows >= cols.
  Index rows = MatrixType::RowsAtCompileTime;
  if (rows == Dynamic) rows = internal::random<Index>(cols, EIGEN_TEST_MAX_SIZE);

  MatrixType A;
  do {
    // To have a unique solution, `A` must be full rank.
    // A Random() matrix might not be full rank, but the probability is very low.
    A = MatrixType::Random(rows, cols);
  } while (A.colPivHouseholderQr().rank() != cols);

  using VectorX = decltype(A.row(0).transpose().eval());
  using VectorB = decltype(A.col(0).eval());
  const VectorX x_expected = VectorX::Random(cols).cwiseAbs();  // non-negative, so it's feasible.
  const VectorB b = A * x_expected;

  test_nnls_known_solution(A, b, x_expected);
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

EIGEN_DECLARE_TEST(NNLS) {
  test_known_problems();
  for (int i = 0; i < g_repeat; i++) {
    test_nnls_dense_solution<MatrixXf>();
    test_nnls_dense_solution<MatrixXd>();
    using MatFixed = Matrix<double, 12, 5>;
    test_nnls_dense_solution<MatFixed>();
  }
}
