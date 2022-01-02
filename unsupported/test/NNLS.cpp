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

  Index max_iter = 5 * cols;  // A heuristic guess.
  NNLS<MatrixType> solver(A, static_cast<int>(max_iter));
  const bool solved = solver.solve(b);
  const VectorX x_solved = solver.x();

  VERIFY(solved);
  VERIFY_IS_APPROX(x_solved, x_expected);
}

EIGEN_DECLARE_TEST(NNLS) {
  for (int i = 0; i < g_repeat; i++) {
    CALL_SUBTEST_1(test_nnls_dense_solution<MatrixXf>());
    CALL_SUBTEST_2(test_nnls_dense_solution<MatrixXd>());

    using Mat12x3 = Matrix<double, 12, 3>;
    CALL_SUBTEST_3(test_nnls_dense_solution<Mat12x3>());
  }
}
