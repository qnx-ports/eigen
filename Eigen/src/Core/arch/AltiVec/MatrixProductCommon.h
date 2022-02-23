//#define EIGEN_POWER_USE_PREFETCH  // Use prefetching in gemm routines
#ifdef EIGEN_POWER_USE_PREFETCH
#define EIGEN_POWER_PREFETCH(p)  prefetch(p)
#else
#define EIGEN_POWER_PREFETCH(p)
#endif

#include "../../InternalHeaderCheck.h"

namespace Eigen {

namespace internal {

template<typename Scalar, typename Packet, typename DataMapper, typename Index, const Index accRows, const Index accCols>
EIGEN_ALWAYS_INLINE void gemm_extra_row(
  const DataMapper& res,
  const Scalar* lhs_base,
  const Scalar* rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index row,
  Index col,
  Index rows,
  Index cols,
  Index remaining_rows,
  const Packet& pAlpha,
  const Packet& pMask);

template<typename Scalar, typename Packet, typename DataMapper, typename Index, const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_STRONG_INLINE void gemm_extra_cols(
  const DataMapper& res,
  const Scalar* blockA,
  const Scalar* blockB,
  Index depth,
  Index strideA,
  Index offsetA,
  Index strideB,
  Index offsetB,
  Index col,
  Index rows,
  Index cols,
  Index remaining_rows,
  const Packet& pAlpha,
  const Packet& pMask);

template<typename Packet, typename Index>
EIGEN_ALWAYS_INLINE Packet bmask(const Index remaining_rows);

template<typename Scalar, typename Packet, typename Packetc, typename DataMapper, typename Index, const Index accRows, const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_ALWAYS_INLINE void gemm_complex_extra_row(
  const DataMapper& res,
  const Scalar* lhs_base,
  const Scalar* rhs_base,
  Index depth,
  Index strideA,
  Index offsetA,
  Index strideB,
  Index row,
  Index col,
  Index rows,
  Index cols,
  Index remaining_rows,
  const Packet& pAlphaReal,
  const Packet& pAlphaImag,
  const Packet& pMask);

template<typename Scalar, typename Packet, typename Packetc, typename DataMapper, typename Index, const Index accCols, bool ConjugateLhs, bool ConjugateRhs, bool LhsIsReal, bool RhsIsReal>
EIGEN_STRONG_INLINE void gemm_complex_extra_cols(
  const DataMapper& res,
  const Scalar* blockA,
  const Scalar* blockB,
  Index depth,
  Index strideA,
  Index offsetA,
  Index strideB,
  Index offsetB,
  Index col,
  Index rows,
  Index cols,
  Index remaining_rows,
  const Packet& pAlphaReal,
  const Packet& pAlphaImag,
  const Packet& pMask);

template<typename Scalar, typename Packet>
EIGEN_ALWAYS_INLINE Packet ploadLhs(const Scalar* lhs);

template<typename DataMapper, typename Packet, typename Index, const Index accCols, int StorageOrder, bool Complex, int N>
EIGEN_ALWAYS_INLINE void bload(PacketBlock<Packet,N*(Complex?2:1)>& acc, const DataMapper& res, Index row, Index col);

template<typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet,N>& acc, PacketBlock<Packet,N>& accZ, const Packet& pAlpha);

template<typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscale(PacketBlock<Packet,N>& acc, PacketBlock<Packet,N>& accZ, const Packet& pAlpha, const Packet& pMask);

template<typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscalec(PacketBlock<Packet,N>& aReal, PacketBlock<Packet,N>& aImag, const Packet& bReal, const Packet& bImag, PacketBlock<Packet,N>& cReal, PacketBlock<Packet,N>& cImag);

template<typename Packet, int N>
EIGEN_ALWAYS_INLINE void bscalec(PacketBlock<Packet,N>& aReal, PacketBlock<Packet,N>& aImag, const Packet& bReal, const Packet& bImag, PacketBlock<Packet,N>& cReal, PacketBlock<Packet,N>& cImag, const Packet& pMask);

template<typename Scalar, typename Packet, typename Index, const Index remaining_rows>
EIGEN_ALWAYS_INLINE void loadPacketRemaining(const Scalar* lhs, Packet &lhsV);

// Grab two decouples real/imaginary PacketBlocks and return two coupled (real/imaginary pairs) PacketBlocks.
template<typename Packet, typename Packetc, int N>
EIGEN_ALWAYS_INLINE void bcouple_common(PacketBlock<Packet,N>& taccReal, PacketBlock<Packet,N>& taccImag, PacketBlock<Packetc, N>& acc1, PacketBlock<Packetc, N>& acc2)
{
  acc1.packet[0].v = vec_mergeh(taccReal.packet[0], taccImag.packet[0]);
  if (N > 1) {
    acc1.packet[1].v = vec_mergeh(taccReal.packet[1], taccImag.packet[1]);
  }
  if (N > 2) {
    acc1.packet[2].v = vec_mergeh(taccReal.packet[2], taccImag.packet[2]);
  }
  if (N > 3) {
    acc1.packet[3].v = vec_mergeh(taccReal.packet[3], taccImag.packet[3]);
  }

  acc2.packet[0].v = vec_mergel(taccReal.packet[0], taccImag.packet[0]);
  if (N > 1) {
    acc2.packet[1].v = vec_mergel(taccReal.packet[1], taccImag.packet[1]);
  }
  if (N > 2) {
    acc2.packet[2].v = vec_mergel(taccReal.packet[2], taccImag.packet[2]);
  }
  if (N > 3) {
    acc2.packet[3].v = vec_mergel(taccReal.packet[3], taccImag.packet[3]);
  }
}

template<typename Packet, typename Packetc, int N>
EIGEN_ALWAYS_INLINE void bcouple(PacketBlock<Packet,N>& taccReal, PacketBlock<Packet,N>& taccImag, PacketBlock<Packetc,N*2>& tRes, PacketBlock<Packetc, N>& acc1, PacketBlock<Packetc, N>& acc2)
{
  bcouple_common<Packet, Packetc, N>(taccReal, taccImag, acc1, acc2);

  acc1.packet[0] = padd<Packetc>(tRes.packet[0], acc1.packet[0]);
  if (N > 1) {
    acc1.packet[1] = padd<Packetc>(tRes.packet[1], acc1.packet[1]);
  }
  if (N > 2) {
    acc1.packet[2] = padd<Packetc>(tRes.packet[2], acc1.packet[2]);
  }
  if (N > 3) {
    acc1.packet[3] = padd<Packetc>(tRes.packet[3], acc1.packet[3]);
  }

  acc2.packet[0] = padd<Packetc>(tRes.packet[0+N], acc2.packet[0]);
  if (N > 1) {
    acc2.packet[1] = padd<Packetc>(tRes.packet[1+N], acc2.packet[1]);
  }
  if (N > 2) {
    acc2.packet[2] = padd<Packetc>(tRes.packet[2+N], acc2.packet[2]);
  }
  if (N > 3) {
    acc2.packet[3] = padd<Packetc>(tRes.packet[3+N], acc2.packet[3]);
  }
}

// This is necessary because ploadRhs for double returns a pair of vectors when MMA is enabled.
template<typename Scalar, typename Packet>
EIGEN_ALWAYS_INLINE Packet ploadRhs(const Scalar* rhs)
{
  return ploadu<Packet>(rhs);
}

#define NEW_EXTRA

#define MICRO_NORMAL(iter) \
  (accCols == accCols2) || (unroll_factor != (iter + 1))

#ifdef NEW_EXTRA
#define MICRO_UNROLL_ITER(func, N) \
  switch (remaining_rows) { \
    default: \
      func(N, 0) \
      break; \
    case 1: \
      func(N, 1) \
      break; \
    case 2: \
      if (sizeof(Scalar) == sizeof(float)) { \
        func(N, 2) \
      } \
      break; \
    case 3: \
      if (sizeof(Scalar) == sizeof(float)) { \
        func(N, 3) \
      } \
      break; \
  }
#else
#define MICRO_UNROLL_ITER(func, N) \
  func(N, 0)
#endif

#define MICRO_LOAD_ONE(iter) \
  if (unroll_factor > iter) { \
    if (MICRO_NORMAL(iter)) { \
      lhsV##iter = ploadLhs<Scalar, Packet>(lhs_ptr##iter); \
      lhs_ptr##iter += accCols; \
    } else { \
      loadPacketRemaining<Scalar, Packet, Index, accCols2>(lhs_ptr##iter, lhsV##iter); \
      lhs_ptr##iter += accCols2; \
    } \
  } else { \
    EIGEN_UNUSED_VARIABLE(lhsV##iter); \
  }

#define MICRO_COMPLEX_LOAD_ONE(iter) \
  if (unroll_factor > iter) { \
    if (MICRO_NORMAL(iter)) { \
      lhsV##iter = ploadLhs<Scalar, Packet>(lhs_ptr_real##iter); \
      if(!LhsIsReal) { \
        lhsVi##iter = ploadLhs<Scalar, Packet>(lhs_ptr_real##iter + imag_delta); \
      } else { \
        EIGEN_UNUSED_VARIABLE(lhsVi##iter); \
      } \
      lhs_ptr_real##iter += accCols; \
    } else { \
      loadPacketRemaining<Scalar, Packet, Index, accCols2>(lhs_ptr_real##iter, lhsV##iter); \
      if(!LhsIsReal) { \
        loadPacketRemaining<Scalar, Packet, Index, accCols2>(lhs_ptr_real##iter + imag_delta2, lhsVi##iter); \
      } else { \
        EIGEN_UNUSED_VARIABLE(lhsVi##iter); \
      } \
      lhs_ptr_real##iter += accCols2; \
    } \
  } else { \
    EIGEN_UNUSED_VARIABLE(lhsV##iter); \
    EIGEN_UNUSED_VARIABLE(lhsVi##iter); \
  }

#define MICRO_SRC_PTR_ONE(iter) \
  if (unroll_factor > iter) { \
    if (MICRO_NORMAL(iter)) { \
      lhs_ptr##iter = lhs_base + (row+(iter*accCols))*strideA; \
    } else { \
      lhs_ptr##iter = lhs_base + (row+(iter*accCols))*strideA - (accCols-accCols2)*offsetA; \
    } \
  } else { \
    EIGEN_UNUSED_VARIABLE(lhs_ptr##iter); \
  }

#define MICRO_COMPLEX_SRC_PTR_ONE(iter) \
  if (unroll_factor > iter) { \
    if (MICRO_NORMAL(iter)) { \
      lhs_ptr_real##iter = lhs_base + (row+(iter*accCols))*strideA*advanceRows; \
    } else { \
      lhs_ptr_real##iter = lhs_base + (row+(iter*accCols))*strideA*advanceRows - (accCols-accCols2)*offsetA; \
    } \
  } else { \
    EIGEN_UNUSED_VARIABLE(lhs_ptr_real##iter); \
  }

} // end namespace internal
} // end namespace Eigen
