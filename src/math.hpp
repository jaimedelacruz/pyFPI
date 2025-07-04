#ifndef MMATHHPP
#define MMATHHPP
/* ---

   Mathematical tools requiered for the FPI transmission profile
   calculations

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)

   --- */


#include <array>
#include <cstring>
#include <algorithm>
#include <complex>
#include <fftw3.h>

namespace mth{

  // ********************************************************************* //

  template<typename T> constexpr
  inline T SQ(T const &var){return var*var;}

  template<typename T> constexpr
  inline T CUB(T const &var){return var*var*var;}
  
  template<typename T> constexpr
  inline T POW4(T const &var){return mth::SQ<T>(mth::SQ<T>(var));}
  
  // ********************************************************************* //

  template<typename T> constexpr
  inline T SignFortran(T const &var){return ((var < T(0))?T(-1) : T(1));}
  
  // ********************************************************************* //

  template<typename T, int N>
  class Linear1D{
    
    /* ---
       1D linear interpolation class
       it stores the interpolation coefficients
       to avoid recomputing them all the time.
       --- */
    
    std::array<T,N> const x;
    std::array<T,N-1> a;
    std::array<T,N-1> b;

  public:
    
    // ------------------------------------------------------ //
    
    Linear1D(std::array<T,N> const& xin, std::array<T,N> const& val):
      x(xin)
    {
      constexpr const int N1 = N-1;
      
      for(int ii=0; ii<N1; ++ii){
	b[ii] = val[ii];
	a[ii] = (val[ii+1] - val[ii]) / (x[ii+1]-x[ii]);
      }
    }
    
    // ------------------------------------------------------ //

    inline void interpolate(int const N1, const T* const xx, T* const yy)const
    {

      // --- checkl if xx is monotonically increasing or decreasing --- //
      
      int i0=0, i1  = N1, di = 1, k = 0;
      
      if((xx[1] - xx[0]) < 0){
	i0 = N1;
	i1 = -1;
	di = -1;
      }

      // --- Now interpolate over all intervals --- //
      
      int const NN = N-1;
      
      for(int ii=i0; ii != i1; ii += di){
	T const ixx = std::min<T>(std::max<T>(x[0], xx[ii]), x[N-1]);
	
	for(int jj=0; jj<NN; ++jj){
	  if(ixx >= x[jj]) k = jj;
	  else break;
	}
	
	yy[ii] = a[k] * (ixx - x[k]) + b[k];
      }
    }
    
    // ------------------------------------------------------ //

    inline T interpolate(T xx)const{
      int const N1 = N-1;

      xx = std::min<T>(std::max<T>(x[0], xx), x[N1]);
      
      int k = 0;
      
      for(int ii=0; ii<N1; ++ii){
	if((xx >= x[ii])) k = ii;
	else break;
      }
      return a[k] * (xx-x[k]) + b[k];
    }

    // ------------------------------------------------------ //

  };
  
  // ********************************************************************* //
  
  /* --- 
     1D FFTW convolution class, useful to perform many convolutions with the
     same PSF (e.g., inversions) because the PSF is only transformed once
     --- */
  
  template <class T>
  struct fftconv1D {
    int npad, n, n1, nft;
    std::complex<double> *otf, *ft;
    fftw_plan fplan, bplan;
    double *padded;
    bool started_plans;
    /* ------------------------------------------------------------------------------- */
    
    fftconv1D():
      npad(0), n(0), n1(0), nft(0), otf(NULL), ft(NULL), fplan(0), \
      bplan(0), padded(NULL), started_plans(false){};
    
    /* ------------------------------------------------------------------------------- */

    fftconv1D(fftconv1D<T> const& in):
      fftconv1D()
    {
      *this = in;
    }

    /* ------------------------------------------------------------------------------- */

    fftconv1D<T> &operator=(fftconv1D<T> const& in){

            
      // -- copy dimensions --- //
      npad = in.npad;
      n    = in.n;
      n1   = in.n1;
      nft  = in.nft;

      // --- allocate pointers --- //
      padded = new double [npad]();
      ft     = new std::complex<double> [nft+2]();
      otf    = new std::complex<double> [nft+2]();

      // --- init plans --- //
      fplan = fftw_plan_dft_r2c_1d(npad, padded, reinterpret_cast<fftw_complex*>(ft), FFTW_MEASURE);
      bplan = fftw_plan_dft_c2r_1d(npad, reinterpret_cast<fftw_complex*>(ft), padded, FFTW_MEASURE);

      started_plans = true;


      // --- copy data --- //
      memcpy(padded, in.padded, npad*sizeof(double));
      memcpy(ft,     in.ft,     nft*sizeof(double));
      memcpy(otf,    in.otf,    nft*sizeof(double));
      
      return *this;
    }

    /* ------------------------------------------------------------------------------- */

  fftconv1D(const int n_in, const int n_psf):
    fftconv1D()
    {
      
      if(n_psf == 0){
	return;
      }
    
      /* --- define dimensions --- */

      n = n_in, n1 = n_psf, npad = ((n1/2)*2 == n1) ? n1+n-1 : n1+n;
      nft = npad/2 + 1;
      

      
      /* --- allocate arrays --- */
      
      double* const ppsf   = new double [npad]();
      padded               = new double [npad]();
      
      //
      ft  = new std::complex<double> [nft+2]();
      otf = new std::complex<double> [nft+2]();

      
      /* --- Init forward and backward plans --- */

      fplan = fftw_plan_dft_r2c_1d(npad, padded, reinterpret_cast<fftw_complex*>(ft), FFTW_MEASURE);
      bplan = fftw_plan_dft_c2r_1d(npad, reinterpret_cast<fftw_complex*>(ft), padded, FFTW_MEASURE);
      started_plans = true;
      
      

      /* --- clean-up --- */
      
      delete [] ppsf;
    }
    /* ------------------------------------------------------------------------------- */
    
    void updatePSF(int const inpsf, const T* const __restrict__ psf)const
    {
      if(inpsf != n1){
	fprintf(stderr,"[error] mth::fftconvol1D::updatePSF: object was initialized with a different number of elements for the PSF (%d != %d), fix your code!\n", inpsf, n1);
	exit(1);
      }
      
      double* const __restrict__ ppsf = new double [npad]();

      double  psf_tot = 1.0;
      //for(int ii=0; ii<n1; ii++) psf_tot += psf[ii];
      psf_tot = 1.0 / (psf_tot * npad);
      
      for(int ii = 0; ii<n1; ii++) ppsf[ii] = (double)psf[ii] * psf_tot;
      std::rotate(&ppsf[0], &ppsf[n1/2], &ppsf[npad]);

      
      /* --- FFT transform psf --- */

      fftw_execute_dft_r2c(fplan, ppsf, reinterpret_cast<fftw_complex*>(otf));

      

      /* --- take the conjugate --- */

      for(int ii=0; ii<nft; ++ii)
	otf[ii] = std::conj(otf[ii]);
      
      delete [] ppsf;

    }
        
    /* ------------------------------------------------------------------------------- */

    ~fftconv1D(){

      if(started_plans){
	//fprintf(stderr,"[info] mth::fftconv1D::~fftconv1D: erasing FFTW-3 plans\n");
	fftw_destroy_plan(fplan);
	fftw_destroy_plan(bplan);
      }
      
      if(ft)  delete [] ft;
      if(otf) delete [] otf;
      if(padded) delete [] padded;

      ft = NULL, otf = NULL, padded = NULL, started_plans = false;
      n = 0, n1 = 0, npad = 0, nft = 0;
    }
  /* ------------------------------------------------------------------------------- */
    
    inline void convolve(int const n_in, T *d)const{

      if(npad == 0){
	return;
      }
      
      if(n_in != n){
	fprintf(stderr, "[error] fftconvol1D::convolve: n_in [%d] != n [%d], not convolving!\n", n_in, n);
	return;
      }

      
      /* --- copy data to padded array --- */

      for(int ii = 0; ii<n; ii++)         padded[ii] = (double)d[ii];
      for(int ii = n; ii<n+n1/2; ii++)    padded[ii] = (double)d[n-1];
      for(int ii = n+n1/2; ii<npad; ii++) padded[ii] = (double)d[0];

      
      /* --- Forward transform --- */

      fftw_execute_dft_r2c(fplan, (double*)padded, reinterpret_cast<fftw_complex*>(ft));

      
      
      /* --- Convolve --- */
      
      for(int ii = 0; ii<nft; ii++)
	ft[ii] *= otf[ii];
      

      
      /* --- Backwards transform --- */

      fftw_execute(bplan);

      

      /* --- Copy back data (inplace) --- */

      for(int ii = 0; ii<n; ii++)
	d[ii] = (T)padded[ii];

    }

    /* ------------------------------------------------------------------------------- */
    
    
  }; // fftconvol1D class
  
  // ********************************************************************* //

  template<typename T> inline
  void interpolation_Linear(int const N,  const T* const __restrict__ x,  const T* const __restrict__ y,
			    int const NN, const T* const __restrict__ xx, T* const __restrict__ yy )
  {
    int const N1 = N-1;
    
    // --- pre-compute derivatives --- //
    
    T* const __restrict__ a = new T[N1]();

    for(int ii=0; ii<N1; ++ii){
      a[ii] = (y[ii+1]-y[ii]) / (x[ii+1] - x[ii]);
    }

    int k = 0;
    for(int ii=0; ii < NN; ++ii){
      T const ixx = std::min<T>(std::max<T>(x[0], xx[ii]), x[N1]);

      for(int jj=k; jj<N1; ++jj){
	if(ixx >= x[jj]) k = jj;
	else break;
      }
	
      yy[ii] = a[k] * (ixx - x[k]) + y[k];
    }
    
    delete [] a;
  }

  // ********************************************************************* //

  
}


#endif
