#include "pocketfft_hdronly.h"
using namespace pocketfft;
using namespace pocketfft::detail;

namespace Eigen {

namespace internal {

template<typename _Scalar>
struct pocket_fft_impl
{
  typedef _Scalar Scalar;
  typedef std::complex<Scalar> Complex;

  inline void fwd(Complex* dst, const Complex* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_{ sizeof(Complex) };
    c2c(shape_, stride_, stride_, axes_, FORWARD, src, dst, static_cast<Scalar>(1));
  }

  inline void fwd(Complex* dst, const Scalar* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_in{ sizeof(Scalar) };
    const stride_t stride_out{ sizeof(Complex) };
    r2c(shape_, stride_in, stride_out, axes_, FORWARD, src, dst, static_cast<Scalar>(1));
  }

  inline void inv(Complex* dst, const Complex* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_{ sizeof(Complex) };
    c2c(shape_, stride_, stride_, axes_, BACKWARD, src, dst, static_cast<Scalar>(1));
  }

  inline void inv(Scalar* dst,  const Complex* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_in{ sizeof(Complex) };
    const stride_t stride_out{ sizeof(Scalar) };
    c2r(shape_, stride_in, stride_out, axes_, BACKWARD, src, dst, static_cast<Scalar>(1));
  }

};

} // namespace internal
} // namespace Eigen