// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Shawn Li <tokinobug@163.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX14/Metaheuristic>

#include <iostream>
using namespace Eigen;
using namespace std;

// this example shows how to use box-constraint types
void test_Box() {
    //[0,1]^50, mutate step = 0.02
  BoxNdS<50, DoubleVectorOption::Eigen, true, DivEncode<0, 1>::code, DivEncode<1, 1>::code, DivEncode<1, 50>::code>
      box0;

  box0.min();

  cout << box0.dimensions() << endl;
  cout << box0.learnRate() << endl;
  cout << "sizeof(box0) = " << sizeof(box0) << endl;// every thing about this boxes are known at compile time, so its size is 1


  // Dynamic dim non-suqare box
  BoxXdN<DoubleVectorOption::Eigen> box;

  // Set the min and max by its non-const-ref to max and min members.
  box.max().setConstant(50, 1, 1.0);
  box.min().setConstant(50, 1, 0.0);

  // 10 dimensional binary box
  BoxNb<10> BNb;
  // Dynamic dimensional binary box
  BoxXb<> BXb;

  BXb.setDimensions(400);

  BNb.max();
  BNb.min();

  BXb.max();
  BXb.min();

  cout << BNb.dimensions();
  cout << BXb.dimensions();

  cout << "sizeof BNb=" << sizeof(BNb) << endl;// Every information of BNb is fixed at compile time so its size is 1
  cout << "sizeof BXb=" << sizeof(BXb) << endl;//BXb 's dimensions is known at runtime so its size is 8 (size_t)
}

int main() {
  test_Box();
  system("pause");
  return 0;
}
