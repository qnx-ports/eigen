How to test Eigen
---

After Eigen is built and installed, all test binaries will be located under `<prefix>/bin/eigen_tests`
(prefix defaults to `/usr/local`). Note that the following test(s) are known to fail:

NonLinearOptimization

This is located under `unsupported/` and is known to behave unreliably even on Linux.

In addition, other tests may randomly fail due to quirks of QNX's default random number generator.
A known good seed to set is `EIGEN_SEED=1711113814`.
