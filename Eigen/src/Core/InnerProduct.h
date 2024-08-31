// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2024 Charlie Schlosser <cs.schlosser@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_INNER_PRODUCT_EVAL_H
#define EIGEN_INNER_PRODUCT_EVAL_H

// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

// recursively searches for the largest simd type that does not exceed Size, or the smallest if no such type exists
template <typename Scalar, int Size, typename Packet = typename packet_traits<Scalar>::type,
          bool Stop =
              (unpacket_traits<Packet>::size <= Size) || is_same<Packet, typename unpacket_traits<Packet>::half>::value>
struct find_inner_product_packet_helper;

template <typename Scalar, int Size, typename Packet>
struct find_inner_product_packet_helper<Scalar, Size, Packet, false> {
  using type = typename find_inner_product_packet_helper<Scalar, Size, typename unpacket_traits<Packet>::half>::type;
};

template <typename Scalar, int Size, typename Packet>
struct find_inner_product_packet_helper<Scalar, Size, Packet, true> {
  using type = Packet;
};

template <typename Scalar, int Size>
struct find_inner_product_packet : find_inner_product_packet_helper<Scalar, Size> {};

template <typename Scalar>
struct find_inner_product_packet<Scalar, Dynamic> {
  using type = typename packet_traits<Scalar>::type;
};

template <typename Lhs, typename Rhs>
struct inner_product_assert {
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Lhs)
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(Rhs)
  EIGEN_STATIC_ASSERT_SAME_VECTOR_SIZE(Lhs, Rhs)
#ifndef EIGEN_NO_DEBUG
  static EIGEN_DEVICE_FUNC void run(const Lhs& lhs, const Rhs& rhs) {
    eigen_assert((lhs.size() == rhs.size()) && "Inner product: lhs and rhs vectors must have same size");
  }
#else
  static EIGEN_DEVICE_FUNC void run(const Lhs&, const Rhs&) {}
#endif
};

template <typename Func, typename Lhs, typename Rhs>
struct inner_product_evaluator {
  static constexpr int LhsFlags = evaluator<Lhs>::Flags, RhsFlags = evaluator<Rhs>::Flags,
                       SizeAtCompileTime = min_size_prefer_fixed(Lhs::SizeAtCompileTime, Rhs::SizeAtCompileTime),
                       LhsAlignment = evaluator<Lhs>::Alignment, RhsAlignment = evaluator<Rhs>::Alignment;

  using Scalar = typename Func::result_type;
  using Packet = typename find_inner_product_packet<Scalar, SizeAtCompileTime>::type;

  static constexpr bool Vectorize =
      bool(LhsFlags & RhsFlags & PacketAccessBit) && Func::PacketAccess &&
      ((SizeAtCompileTime == Dynamic) || (unpacket_traits<Packet>::size <= SizeAtCompileTime));

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE explicit inner_product_evaluator(const Lhs& lhs, const Rhs& rhs,
                                                                         Func func = Func())
      : m_func(func), m_lhs(lhs), m_rhs(rhs), m_size(lhs.size()) {
    inner_product_assert<Lhs, Rhs>::run(lhs, rhs);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Index size() const { return m_size.value(); }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& value, Index index) const {
    return m_func.coeff(value, m_lhs.coeff(index), m_rhs.coeff(index));
  }

  template <typename PacketType, int LhsMode = LhsAlignment, int RhsMode = RhsAlignment>
  EIGEN_STRONG_INLINE PacketType packet(PacketType value, Index index) const {
    return m_func.packet(value, m_lhs.template packet<LhsMode, PacketType>(index),
                         m_rhs.template packet<RhsMode, PacketType>(index));
  }

  const Func m_func;
  const evaluator<Lhs> m_lhs;
  const evaluator<Rhs> m_rhs;
  const variable_if_dynamic<Index, SizeAtCompileTime> m_size;
};

template <typename Evaluator, bool Vectorize = Evaluator::Vectorize>
struct inner_product_impl;

// scalar loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, false> {
  using Scalar = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const Index size = eval.size();
    const Index size2 = numext::round_down(size, 2);
    Scalar result = Scalar(0);
    if (size2 > 0) {
      Scalar result2 = Scalar(0);
      for (Index k = 0; k < size2; k += 2) {
        result = eval.coeff(result, k);
        result2 = eval.coeff(result2, k + 1);
      }
      result += result2;
    }
    if (size > size2) result = eval.coeff(result, size2);
    return result;
  }
};

// vector loop
template <typename Evaluator>
struct inner_product_impl<Evaluator, true> {
  using Scalar = typename Evaluator::Scalar;
  using Packet = typename Evaluator::Packet;
  static constexpr int PacketSize = unpacket_traits<Packet>::size;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar run(const Evaluator& eval) {
    const Index size = eval.size();
    const Index packetEnd = numext::round_down(size, PacketSize);
    const Index packetEnd2 = numext::round_down(size, 2 * PacketSize);
    Scalar result = Scalar(0);
    if (packetEnd > 0) {
      Packet presult = pzero(Packet());
      if (packetEnd2 > 0) {
        Packet presult2 = pzero(Packet());
        for (Index k = 0; k < packetEnd2; k += 2 * PacketSize) {
          presult = eval.packet(presult, k);
          presult2 = eval.packet(presult2, k + PacketSize);
        }
        presult = padd(presult, presult2);
      }
      if (packetEnd > packetEnd2) {
        presult = eval.packet(presult, packetEnd2);
      }
      result = predux(presult);
    }
    for (Index k = packetEnd; k < size; k++) result = eval.coeff(result, k);
    return result;
  }
};

template <typename Scalar, bool Conj>
struct conditional_conj;

template <typename Scalar>
struct conditional_conj<Scalar, true> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a) { return numext::conj(a); }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a) {
    return pconj(a);
  }
};

template <typename Scalar>
struct conditional_conj<Scalar, false> {
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& a) { return a; }
  template <typename Packet>
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& a) {
    return a;
  }
};

template <typename LhsScalar, typename RhsScalar, bool Conj>
struct scalar_inner_product_op {
  using result_type = typename ScalarBinaryOpTraits<LhsScalar, RhsScalar>::ReturnType;
  using conj_helper = conditional_conj<LhsScalar, Conj>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type coeff(const result_type& accum, const LhsScalar& a,
                                                          const RhsScalar& b) const {
    return (conj_helper::coeff(a) * b) + accum;
  }
  static constexpr bool PacketAccess = false;
};

template <typename Scalar, bool Conj>
struct scalar_inner_product_op<Scalar, Scalar, Conj> {
  using result_type = Scalar;
  using conj_helper = conditional_conj<Scalar, Conj>;
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar coeff(const Scalar& accum, const Scalar& a, const Scalar& b) const {
    return pmadd(conj_helper::coeff(a), b, accum);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packet(const Packet& accum, const Packet& a, const Packet& b) const {
    return pmadd(conj_helper::packet(a), b, accum);
  }
  static constexpr bool PacketAccess = packet_traits<Scalar>::HasMul && packet_traits<Scalar>::HasAdd;
};

template <typename Lhs, typename Rhs, bool Conj>
struct default_inner_product_impl {
  using LhsScalar = typename traits<Lhs>::Scalar;
  using RhsScalar = typename traits<Rhs>::Scalar;
  using Op = scalar_inner_product_op<LhsScalar, RhsScalar, Conj>;
  using Evaluator = inner_product_evaluator<Op, Lhs, Rhs>;
  using result_type = typename Evaluator::Scalar;
  static EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE result_type run(const MatrixBase<Lhs>& a, const MatrixBase<Rhs>& b) {
    Evaluator eval(a.derived(), b.derived(), Op());
    return inner_product_impl<Evaluator>::run(eval);
  }
};

template <typename Lhs, typename Rhs>
struct dot_impl : default_inner_product_impl<Lhs, Rhs, true> {};

}  // namespace internal
}  // namespace Eigen

#endif  // EIGEN_INNER_PRODUCT_EVAL_H
