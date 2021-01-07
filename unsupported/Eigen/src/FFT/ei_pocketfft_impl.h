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

  inline void fwd(Complex* dst, Complex* src, int nfft){

  }

  inline void fwd(Complex* dst, Scalar* src, int nfft){
    
  }
}

} // namespace internal
} // namespace Eigen