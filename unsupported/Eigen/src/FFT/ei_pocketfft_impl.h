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
    const stride_t stride_{ 1 };
    c2c(shape_, stride_, stride_, axes_, FORWARD, src, dst, static_cast<Scalar>(nfft));
  }

  inline void fwd(Complex* dst, const Scalar* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_{ 1 };
    r2c(shape_, stride_, stride_, axes_, FORWARD, src, dst, static_cast<Scalar>(nfft));
  }

  inline void inv(Complex* dst, const Complex* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_{ 1 };
    c2c(shape_, stride_, stride_, axes_, BACKWARD, src, dst, static_cast<Scalar>(nfft));
  }

  inline void inv(Scalar* dst,  const Complex* src, int nfft){
    const shape_t  shape_{ static_cast<size_t>(nfft) };
    const shape_t  axes_{ 0 };
    const stride_t stride_{ 1 };
    c2r(shape_, stride_, stride_, axes_, BACKWARD, src, dst, static_cast<Scalar>(nfft));
  }

};

} // namespace internal
} // namespace Eigen