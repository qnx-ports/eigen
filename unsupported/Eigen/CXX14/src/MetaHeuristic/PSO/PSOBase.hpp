// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Shawn Li <tokinobug@163.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HEU_PSOBASE_HPP
#define EIGEN_HEU_PSOBASE_HPP

#include "InternalHeaderCheck.h"
#include "PSOOption.hpp"
#include "PSOAbstrcat.hpp"

namespace Eigen {

namespace internal {

// Abstrcat base class for most PSO solvers

/**
 * \ingroup CXX14_METAHEURISTIC
 * \class PSOBase
 * \brief Internal base class for PSO solvers
 *
 * This class implements PSO solvers with fixed dimensions
 *
 * \tparam Var_t
 * \tparam DIM Dimension of Var_t
 * \tparam Fitness_t
 * \tparam Record
 * \tparam Arg_t
 * \tparam _iFun_
 * \tparam _fFun_
 *
 * \sa internal::PSOBase<Var_t,Eigen::Dynamic,Fitness_t,Record,Arg_t,_iFun_,_fFun_> for specialization for dynamic-dims.
 */
template <class Var_t, int DIM, class Fitness_t, RecordOption Record, class Arg_t,
          typename PSOParameterPack<Var_t, Fitness_t, Arg_t>::iFun_t _iFun_,
          typename PSOParameterPack<Var_t, Fitness_t, Arg_t>::fFun_t _fFun_>
class PSOBase : public PSOAbstract<Var_t, Fitness_t, Record, Arg_t, _iFun_, _fFun_> {
  using Base_t = PSOAbstract<Var_t, Fitness_t, Record, Arg_t, _iFun_, _fFun_>;

 public:
  virtual ~PSOBase() {}

  EIGEN_HEU_MAKE_PSOABSTRACT_TYPES(Base_t)

  /**
   * \brief Get dimensions
   *
   * \return constexpr int Dimensions at compile time
   */
  constexpr int dimensions() const { return DIM; }

 protected:
  /// Compile time dimensions
  static const constexpr int dims = DIM;
};

//

/**
 * \ingroup CXX14_METAHEURISTIC
 * \class PSOBase<Var_t, Eigen::Dynamic, Fitness_t, Record, Arg_t, _iFun_, _fFun_>
 * \brief Partial specialization for PSOBase with Runtime dimensions
 *
 * \sa PSOBase
 *
 * \tparam Var_t
 * \tparam Fitness_t
 * \tparam Record
 * \tparam Arg_t
 * \tparam _iFun_
 * \tparam _fFun_
 */
template <class Var_t, class Fitness_t, RecordOption Record, class Arg_t,
          typename PSOParameterPack<Var_t, Fitness_t, Arg_t>::iFun_t _iFun_,
          typename PSOParameterPack<Var_t, Fitness_t, Arg_t>::fFun_t _fFun_>
class PSOBase<Var_t, Eigen::Dynamic, Fitness_t, Record, Arg_t, _iFun_, _fFun_>
    : public PSOAbstract<Var_t, Fitness_t, Record, Arg_t, _iFun_, _fFun_> {
  using Base_t = PSOAbstract<Var_t, Fitness_t, Record, Arg_t, _iFun_, _fFun_>;

 public:
  virtual ~PSOBase() {}
  EIGEN_HEU_MAKE_PSOABSTRACT_TYPES(Base_t)

  /**
   * \brief Get dimensions at runtime
   *
   * Runtime assertion will fail if the size of `this->_posMin`, `this->_posMax` and `this->_velocityMax` doesn't match.
   *
   * \return int Dimensions
   */
  inline int dimensions() const {
    assert(this->_posMin.size() == this->_posMax.size());
    assert(this->_posMax.size() == this->_velocityMax.size());
    return this->_posMin.size();
  }

  /**
   * \brief Set the Dimensions
   *
   * This function resizes `this->_posMin`, `this->_posMax` and `this->_velocityMax` to d.
   * \param d Dimension
   */
  inline void setDimensions(int d) {
    this->_posMin.resize(d);
    this->_posMax.resize(d);
    this->_velocityMax.resize(d);
  }
};

}  //  namespace internal

}  //  namespace Eigen
#endif  // EIGEN_HEU_PSOBASE_HPP