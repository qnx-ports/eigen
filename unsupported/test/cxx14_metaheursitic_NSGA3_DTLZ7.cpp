// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Shawn Li <tokinobug@163.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#define _USE_MATH_DEFINES
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX14/Metaheuristic>
#include <iostream>
#include <ctime>
using namespace Eigen;
using namespace std;

// DTLZ 7 is a testing function for many objective problems. Its Pareto frontier consists of many separated parts, which
// examine the algorithm's ability to maintain the population is several niches.
template <size_t M, size_t N>
void DTLZ7(const Eigen::Array<double, N, 1>* x, Eigen::Array<double, M, 1>* f) {
  static_assert(M >= 2, "actural objective amount mustn't be less than 2");
  f->resize(M, 1);
  auto xm = x->template bottomRows<N - M + 1>();
  f->template topRows<M - 1>() = x->template topRows<M - 1>();
  const double g = 1 + 9 * (xm.sum()) /
                           // std::sqrt(1e-40+xm.square().sum())
                           (N - M + 1);
  auto fi = f->template topRows<M - 1>();
  auto one_add_sum3pifi = 1 + (3 * M_PI * fi).sin();
  const double h = M - (fi * one_add_sum3pifi / (1 + g)).sum();
  f->operator[](M - 1) = (1 + g) * h;
}

// DTLZ1 is also provided here
template <size_t M, size_t N>
void DTLZ1(const Eigen::Array<double, N, 1>* x, Eigen::Array<double, M, 1>* f) {
  f->resize(M, 1);
  auto xm = x->template bottomRows<N - M + 1>();
  double g = (xm - 0.5).square().sum() - (20 * M_PI * (xm - 0.5)).cos().sum();
  g += N - M + 1;
  g *= 100;

  double accum = (1 + g) * 0.5;

  int i = 0;
  for (int obj = M - 1; obj > 0; obj--) {
    f->operator[](obj) = accum * (1 - x->operator[](i));
    accum *= x->operator[](i);
    i++;
  }
  f->operator[](0) = x->template topRows<M - 1>().prod() * (1 + g) / 2;
}

void testNSGA3_DTLZ7() {
  // Dimension of decison variable
  static const size_t N = 20;
  // Number of objectives. DTLZ series are greatly extentable, you can simply edit the value of M and N to see how it
  // behaves for higher dimensions.
  static const size_t M = 6;

  // Note that NSGA3 supports only FITNESS_LESS_BETTER
  using solver_t = NSGA3<Eigen::Array<double, N, 1>, M,
                         // FITNESS_LESS_BETTER,
                         DONT_RECORD_FITNESS, SINGLE_LAYER, void>;

  using Var_t = Eigen::Array<double, N, 1>;
  // using Fitness_t = solver_t::Fitness_t;

  auto iFun = GADefaults<Var_t, void, ContainerOption::Eigen>::iFunNd<>;

  auto cFun = GADefaults<Var_t, void, ContainerOption::Eigen>::cFunNd<DivEncode<1, 10>::code>;

  auto mFun = [](const Var_t* src, Var_t* v) {
    *v = *src;
    const size_t idx = ei_randIdx(v->size());
    double& p = v->operator[](idx);
    p += 0.5 * ei_randD(-1, 1);
    p = std::min(p, 1.0);
    p = std::max(p, 0.0);
  };

  GAOption opt;
  cout << "maxGenerations=";
  cin >> opt.maxGenerations;

  // opt.maxFailTimes = -1; // this member is useless to MOGA solvers
  cout << "populationSize=";
  cin >> opt.populationSize;
  opt.crossoverProb = 0.8;
  opt.mutateProb = 0.1;

  solver_t solver;
  // solver.setObjectiveNum(M);
  solver.setiFun(iFun);
  solver.setfFun(DTLZ7<M, N>);
  solver.setcFun(cFun);
  solver.setmFun(mFun);
  solver.setOption(opt);
  solver.setReferencePointPrecision(10);
  cout << "RPCount=" << solver.referencePointCount() << endl;
  solver.initializePop();

  cout << "RP=[" << solver.referencePoints() << "]';\n\n\n" << endl;

  clock_t c = clock();
  solver.run();
  c = clock() - c;

  cout << "solving finished in " << c << " ms with " << solver.generation() << " generations." << endl;

  cout << "PFV=[";
  for (const auto& i : solver.pfGenes()) {
    cout << i->_Fitness.transpose() << ";\n";
  }
  cout << "];\n\n\n" << endl;
}

// This function goes through the whole high-dimensional searching space to find an accurate solution of PF. It's really
// slow but you can find if the solution of NSGA3 is correct.
template <size_t M, size_t N, size_t precision>
void searchPFfun(size_t nIdx, Eigen::Array<double, N, 1>* var, vector<pair<Eigen::Array<double, M, 1>, size_t>>* dst) {
  static_assert(precision >= 2, "You should assign at least 2 on a single dim");
  if (nIdx + 1 >= N) {
    Eigen::Array<double, M, 1> f;
    for (size_t i = 0; i <= precision; i++) {
      var->operator[](nIdx) = double(i) / precision;
      DTLZ7<M, N>(var, &f);
      dst->emplace_back(make_pair(f, 0ULL));
    }
    return;
  }

  for (size_t i = 0; i <= precision; i++) {
    var->operator[](nIdx) = double(i) / precision;
    searchPFfun<M, N, precision>(nIdx + 1, var, dst);
  }
}

int main() {
  testNSGA3_DTLZ7();
  system("pause");
  return 0;
}