// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Shawn Li <tokinobug@163.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_HEU_NSGA3_HPP
#define EIGEN_HEU_NSGA3_HPP

#include "InternalHeaderCheck.h"
#include "NSGA3Base.hpp"

namespace Eigen {

/**
 * @brief Parital specialization for NSGA3 using Eigen's array as fitness values
 */

/**
 * \ingroup CXX14_METAHEURISTIC
 * \brief NSGA3 is the thrid Nondominated Sorting Genetic Algorithm. It's suitable for many objective problems.
 *
 * NSGA3 uses many reference points to maintain a diverse and uniform PF.
 * \sa internal::NSGA3Abstract::select for this special procedure.
 *
 *
 * The default value of template parameters are listed in braces
 * \tparam Var_t Type of decision variable
 * \tparam ObjNum Number of objectives (Eigen::Dynamic for runtime)
 * \tparam rOpt Record fitness or not (don't record)
 * \tparam rpOpt Reference point option (Single layer)
 * \tparam Args_t Other parameters (void)
 * \tparam _iFun_ Initialization function (nullptr)
 * \tparam _fFun_ Fitness function (nullptr)
 * \tparam _cFun_ Crossover function (nullptr)
 * \tparam _mFun_ Mutation function (nullptr)
 *
 * \sa SOGA for APIs that all genetic solvers have
 * \sa NSGA2 for APIs that all MOGA solvers have
 *
 * ## APIs that all NSGA3 solvers have:
 * - `const RefMat_t& referencePoints() const` returns a matrix of reference points. Each coloumn is the coordinate of a
 * RP. (Here RP refers to the word reference point)
 * - `size_t referencePointCount() const` number of reference points according to the RP precision.
 *
 *
 * ## APIs that NSGA3 solvers using single-layer RPs have:
 * - `size_t referencePointPrecision() const` returns the RP precision.
 * - `void setReferencePointPrecision(size_t)` set the RP precision
 *
 *
 * ## APIs that NSGA2 solvers using double-layer RPs have:
 * - `size_t innerPrecision() const` returns the precision of inner layer RP.
 * - `size_t outerPrecision() const` returns the precision of outer layer RP.
 * - `void setReferencePointPrecision(size_t i, size_t o)` set the precison of inner and outer layer.
 *
 * \note When using NSGA3 solvers, is strongly recommended to set the precision explicitly before initializing the
 * population. Don't rely on the default value!
 */
template <typename Var_t, int ObjNum, RecordOption rOpt = DONT_RECORD_FITNESS,
          ReferencePointOption rpOpt = ReferencePointOption::SINGLE_LAYER, class Args_t = void,
          typename internal::GAAbstract<Var_t, Eigen::Array<double, ObjNum, 1>, Args_t>::initializeFun _iFun_ = nullptr,
          typename internal::GAAbstract<Var_t, Eigen::Array<double, ObjNum, 1>, Args_t>::fitnessFun _fFun_ = nullptr,
          typename internal::GAAbstract<Var_t, Eigen::Array<double, ObjNum, 1>, Args_t>::crossoverFun _cFun_ = nullptr,
          typename internal::GAAbstract<Var_t, Eigen::Array<double, ObjNum, 1>, Args_t>::mutateFun _mFun_ = nullptr>
class NSGA3 : public internal::NSGA3Base<Var_t, ObjNum, rOpt, rpOpt, Args_t, _iFun_, _fFun_, _cFun_, _mFun_> {
  using Base_t = internal::NSGA3Base<Var_t, ObjNum, rOpt, rpOpt, Args_t, _iFun_, _fFun_, _cFun_, _mFun_>;

 public:
  NSGA3() {}
  virtual ~NSGA3() {}
  EIGEN_HEU_MAKE_NSGA3ABSTRACT_TYPES(Base_t)

  /**
   * \brief
   *
   */
  void initializePop() {
    this->makeReferencePoses();
    Base_t::initializePop();
  }
};

}  //  namespace Eigen

#endif  //  EIGEN_HEU_NSGA3_HPP