#Eigen

**Eigen is a C++ template library for linear algebra: matrices, vectors, numerical solvers, and related algorithms.**

For more information go to http://eigen.tuxfamily.org/.

For ***pull request***, ***bug reports***, and ***feature requests***, go to https://gitlab.com/libeigen/eigen.


## Building
This code base uses CMake as its core build system. The following options can be used to control the behavior of the build

### Default Build Example
``
   mkdir build
   cd build
   cmake ..
``

### Header Only Installation Example
```
   mkdir build-headeronly
   cd build-headeronly
   cmake .. -DEIGEN_BUILD_BTL=OFF -DEIGEN_BUILD_DOC=OFF  \
            -DEIGEN_BUILD_BLAS=OFF -DEIGEN_BUILD_LAPACK=OFF \
            -DEIGEN_BUILD_PKGCONFIG=ON -DEIGEN_BUILD_DOC=OFF \
            -DEIGEN_BUILD_TESTING=OFF
```
### Options
|Option | Descirption |  Default|
|-------|-------------|---------|
| EIGEN_BUILD_DOC | Enable creation of Eigen documentation | ON |
| EIGEN_BUILD_TESTING | Enable creation of Eigen tests. | ON |
| EIGEN_BUILD_BLAS | Toogles on the building of the Eigen Blas library | ON |
| EIGEN_BUILD_LAPACK | Toggles the building of the included Eigen LAPACK library | ON |
| EIGEN_BUILD_BTL | Build benchmark suite | OFF |
| EIGEN_BUILD_PKGCONFIG | Build pkg-config .pc file for Eigen | ON (!WIN32) |
| CMAKE_BUILD_EXPORT_TARGETS | Controls the installation of cmake helper files find_package commands of downstream projects. | OFF |
| EIGEN_TEST_CXX11 | Enable testing with C++11 and C++11 features (e.g. Tensor module). | OFF |
| EIGEN_SPLIT_LARGE_TESTS | Split large tests into smaller executables | ON |
| EIGEN_DEFAULT_TO_ROW_MAJOR | Use row-major as default matrix storage order | OFF |
| EIGEN_TEST_SSE2 | Enable/Disable SSE2 in tests/examples | OFF |
| EIGEN_TEST_SSE3 | Enable/Disable SSE3 in tests/examples | OFF |
| EIGEN_TEST_SSSE3 | Enable/Disable SSSE3 in tests/examples | OFF |
| EIGEN_TEST_SSE4_1 | Enable/Disable SSE4.1 in tests/examples | OFF |
| EIGEN_TEST_SSE4_2 | Enable/Disable SSE4.2 in tests/examples | OFF |
| EIGEN_TEST_AVX | Enable/Disable AVX in tests/examples | OFF |
| EIGEN_TEST_FMA | Enable/Disable FMA in tests/examples | OFF |
| EIGEN_TEST_AVX2 | Enable/Disable AVX2 in tests/examples | OFF |
| EIGEN_TEST_AVX512 | Enable/Disable AVX512 in tests/examples | OFF |
| EIGEN_TEST_AVX512DQ | Enable/Disable AVX512DQ in tests/examples | OFF |
| EIGEN_TEST_F16C | Enable/Disable F16C in tests/examples | OFF |
| EIGEN_TEST_ALTIVEC | Enable/Disable AltiVec in tests/examples | OFF |
| EIGEN_TEST_VSX | Enable/Disable VSX in tests/examples | OFF |
| EIGEN_TEST_MSA | Enable/Disable MSA in tests/examples | OFF |
| EIGEN_TEST_NEON | Enable/Disable Neon in tests/examples | OFF |
| EIGEN_TEST_NEON64 | Enable/Disable Neon in tests/examples | OFF |
| EIGEN_TEST_Z13 | Enable/Disable S390X(zEC13) ZVECTOR in tests/examples | OFF |
| EIGEN_TEST_Z14 | Enable/Disable S390X(zEC14) ZVECTOR in tests/examples | OFF |
| EIGEN_TEST_OPENMP | Enable/Disable OpenMP in tests/examples | OFF |
| EIGEN_TEST_OPENMP | Enable/Disable OpenMP in tests/examples | OFF |
| EIGEN_TEST_SSE2 | Enable/Disable SSE2 in tests/examples | OFF |
| EIGEN_TEST_AVX | Enable/Disable AVX in tests/examples | OFF |
| EIGEN_TEST_FMA | Enable/Disable FMA/AVX2 in tests/examples | OFF |
| EIGEN_TEST_NO_EXPLICIT_VECTORIZATION | Disable explicit vectorization in tests/examples | OFF |
| EIGEN_TEST_X87 | Force using X87 instructions. Implies no vectorization. | OFF |
| EIGEN_TEST_32BIT | Force generating 32bit code. | OFF |
| EIGEN_TEST_NO_EXPLICIT_ALIGNMENT | Disable explicit alignment (hence vectorization) in tests/examples | OFF |
| EIGEN_TEST_NO_EXCEPTIONS | Disables C++ exceptions | OFF |
| EIGEN_TEST_SYCL | Add Sycl support. | OFF |
| EIGEN_SYCL_TRISYCL | Use the triSYCL Sycl implementation (ComputeCPP by default). | OFF |
| COMPUTECPP_USE_COMPILER_DRIVE | Use ComputeCpp driver instead of a 2 steps compilation | OFF |
| EIGEN_DONT_VECTORIZE_SYCL | Don't use vectorisation in the SYCL tests. | OFF |
