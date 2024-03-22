Building Eigen
---

Makefiles under `qnx/build` are only required to build Eigen's test binaries.
These Makefiles also install ported Eigen headers, but they are not required and you could
simply include / vendor this entire repository for those headers.

Before building Eigen and its tests, you might want to first build and install `muslflt`
under the same staging directory. Projects using Eigen on QNX might also want to link to
`muslflt` for consistent math behavior as other platforms. Without `muslflt`, some tests
may fail and you may run into inconsistencies in results compared to other platforms.

To use these Makefiles:

1. Source `qxnsdp-env.sh` from your QNX SDP
2. Optionally set up a staging area folder (e.g. `/tmp/staging`)
3. `make -C qnx/build INSTALL_ROOT_nto=/tmp/staging USE_INSTALL_ROOT=true JLEVEL=$(nproc) install`
