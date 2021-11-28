/*
 Copyright (c) 2011, Intel Corporation. All rights reserved.

 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to LAPACKe
 *     LU decomposition with partial pivoting based on LAPACKE_?getrf function.
 ********************************************************************************
*/

#ifndef EIGEN_PARTIALLU_LAPACK_H
#define EIGEN_PARTIALLU_LAPACK_H

#include "./InternalHeaderCheck.h"

namespace Eigen { 

namespace internal {

namespace lapacke_helpers {
// -------------------------------------------------------------------------------------------------------------------
//        Dispatch for getrf handling double, float, complex double, complex float types
// -------------------------------------------------------------------------------------------------------------------
lapack_int getrf(lapack_int matrix_order, lapack_int m, lapack_int n, float* a, lapack_int lda, lapack_int *ipiv) {
  return LAPACKE_sgetrf(matrix_order, m, n, a, lda, ipiv);
}
lapack_int getrf(lapack_int matrix_order, lapack_int m, lapack_int n, double* a, lapack_int lda, lapack_int *ipiv) {
  return LAPACKE_dgetrf(matrix_order, m, n, a, lda, ipiv);
}
lapack_int getrf(lapack_int matrix_order, lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int *ipiv) {
  return LAPACKE_zgetrf(matrix_order, m, n, a, lda, ipiv);
}
lapack_int getrf(lapack_int matrix_order, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int *ipiv) {
  return LAPACKE_cgetrf(matrix_order, m, n, a, lda, ipiv);
}

// -------------------------------------------------------------------------------------------------------------------
//        Generic lapacke partial lu implementation that converts arguments and dispatches to the function above
// -------------------------------------------------------------------------------------------------------------------

template<typename Scalar, int StorageOrder>
struct lapacke_partial_lu {
  using BlasType = typename translate_type<Scalar>::type;

  /** \internal performs the LU decomposition in-place of the matrix represented */
  static lapack_int blocked_lu(Index rows, Index cols, Scalar* lu_data, Index luStride, lapack_int* row_transpositions,
  lapack_int& nb_transpositions, lapack_int maxBlockSize=256)
  {
    EIGEN_UNUSED_VARIABLE(maxBlockSize);
    // Set up parameters for getrf
    lapack_int matrix_order = StorageOrder==RowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR;
    lapack_int lda = convert_index<lapack_int>(luStride);
    Scalar* a = lu_data;
    lapack_int* ipiv = row_transpositions;
    lapack_int m = convert_index<lapack_int>(rows);
    lapack_int n = convert_index<lapack_int>(cols);
    nb_transpositions = 0;

    lapack_int info = getrf( matrix_order, m, n, (BlasType*)a, lda, ipiv );
    eigen_assert(info >= 0);

    for(int i=0; i<m; i++) {
      ipiv[i]--;
      if (ipiv[i] != i) nb_transpositions++;
    }
    lapack_int first_zero_pivot = info;
    return first_zero_pivot;
  }
};
} // end namespace lapacke_helpers

/*
 * Here, we just put the generic implementation from lapacke_partial_lu into a partial specialization of the partial_lu_impl
 * type. This specialization is more specialized than the generic implementations that Eigen implements, so if the
 * Scalar type matches they will be chosen.
 */
template<int StorageOrder>
struct partial_lu_impl<double, StorageOrder, lapack_int, Dynamic> : public lapacke_helpers::lapacke_partial_lu<double, StorageOrder> {};

template<int StorageOrder>
struct partial_lu_impl<float, StorageOrder, lapack_int, Dynamic> : public lapacke_helpers::lapacke_partial_lu<float, StorageOrder> {};

template<int StorageOrder>
struct partial_lu_impl<dcomplex, StorageOrder, lapack_int, Dynamic> : public lapacke_helpers::lapacke_partial_lu<dcomplex, StorageOrder> {};

template<int StorageOrder>
struct partial_lu_impl<scomplex, StorageOrder, lapack_int, Dynamic> : public lapacke_helpers::lapacke_partial_lu<scomplex, StorageOrder> {};

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_PARTIALLU_LAPACK_H
