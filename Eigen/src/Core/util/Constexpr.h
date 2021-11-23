// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2021 Erik Schultheis <erik.schultheis@aalto.fi>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

/**
 * \file This file defines common constexpr functions. It is included after `Macros.h` and `Constants.h` so it can make
 * use of these, but before any actual source code.
 */

#ifndef EIGEN_CONSTEXPR_H
#define EIGEN_CONSTEXPR_H

#include "../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {
  /// \internal Returns true if its argument is of integer or enum type.
  /// FIXME this has the same purpose as `is_valid_index_type` in XprHelper.h
  template<typename A>
  constexpr bool is_int_or_enum = std::is_enum<A>::value || std::is_integral<A>::value;
}

/// \internal Gets the minimum of two values which may be integers or enums
template<class A, class B>
inline constexpr int EIGEN_PLAIN_ENUM_MIN(A a, B b) {
  static_assert(internal::is_int_or_enum<A>, "Argument a must be an integer or enum");
  static_assert(internal::is_int_or_enum<B>, "Argument b must be an integer or enum");
  return ((int) a <= (int) b) ? (int)a : (int)b;
}

/// \internal Gets the maximum of two values which may be integers or enums
template<class A, class B>
inline constexpr int EIGEN_PLAIN_ENUM_MAX(A a, B b) {
  static_assert(internal::is_int_or_enum<A>, "Argument a must be an integer or enum");
  static_assert(internal::is_int_or_enum<B>, "Argument b must be an integer or enum");
  return ((int) a >= (int) b) ? (int) a : (int) b;
}

/**
 * \internal
 *  EIGEN_SIZE_MIN_PREFER_DYNAMIC gives the min between compile-time sizes. 0 has absolute priority, followed by 1,
 *  followed by Dynamic, followed by other finite values. The reason for giving Dynamic the priority over
 *  finite values is that min(3, Dynamic) should be Dynamic, since that could be anything between 0 and 3.
 */
template<class A, class B>
inline constexpr int EIGEN_SIZE_MIN_PREFER_DYNAMIC(A a, B b) {
  static_assert(internal::is_int_or_enum<A>, "Argument a must be an integer or enum");
  static_assert(internal::is_int_or_enum<B>, "Argument b must be an integer or enum");
  if ((int) a == 0 || (int) b == 0) return 0;
  if ((int) a == 1 || (int) b == 1) return 1;
  if ((int) a == Dynamic || (int) b == Dynamic) return Dynamic;
  return EIGEN_PLAIN_ENUM_MIN(a, b);
}

/**
 * \internal
 *  EIGEN_SIZE_MIN_PREFER_FIXED is a variant of EIGEN_SIZE_MIN_PREFER_DYNAMIC comparing MaxSizes. The difference is that finite values
 *  now have priority over Dynamic, so that min(3, Dynamic) gives 3. Indeed, whatever the actual value is
 *  (between 0 and 3), it is not more than 3.
 */
template<class A, class B>
inline constexpr int EIGEN_SIZE_MIN_PREFER_FIXED(A a, B b) {
  static_assert(internal::is_int_or_enum<A>, "Argument a must be an integer or enum");
  static_assert(internal::is_int_or_enum<B>, "Argument b must be an integer or enum");
  if ((int) a == 0 || (int) b == 0) return 0;
  if ((int) a == 1 || (int) b == 1) return 1;
  if ((int) a == Dynamic && (int) b == Dynamic) return Dynamic;
  if ((int) a == Dynamic) return (int) b;
  if ((int) b == Dynamic) return (int) a;
  return EIGEN_PLAIN_ENUM_MIN(a, b);
}

/// \internal see EIGEN_SIZE_MIN_PREFER_DYNAMIC. No need for a separate variant for MaxSizes here.
template<class A, class B>
inline constexpr int EIGEN_SIZE_MAX(A a, B b) {
  static_assert(internal::is_int_or_enum<A>, "Argument a must be an integer or enum");
  static_assert(internal::is_int_or_enum<B>, "Argument b must be an integer or enum");
  if ((int) a == Dynamic || (int) b == Dynamic) return Dynamic;
  return EIGEN_PLAIN_ENUM_MAX(a, b);
}

/// \internal Calculate logical XOR at compile time
inline constexpr bool EIGEN_LOGICAL_XOR(bool a, bool b) {
  return (a || b) && !(a && b);
}

/// \internal Calculate logical IMPLIES at compile time
inline constexpr bool EIGEN_IMPLIES(bool a, bool b) {
  return !a || b;
}

}

#endif // EIGEN_CONSTEXPR_H