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

  template<typename T>
  inline std::array<T,3> parab_fit(const T* const x, const T* const y){

    std::array<T,3> cf;
    
    T const d = x[0];
    T const e = x[1];
    T const f = x[2];
  
    T const yd = y[0];
    T const ye = y[1];
    T const yf = y[2];
  
    cf[1] = ((yf - yd) - (mth::SQ(f) - mth::SQ(d)) * ((ye - yd) / (mth::SQ(e) - mth::SQ(d))))/ \
      ((f - d) - (mth::SQ(f) - mth::SQ(d)) * ((e - d) / (mth::SQ(e) - mth::SQ(d))));
    
    cf[2] = ((ye - yd) - cf[1] * (e - d)) / (mth::SQ(e) - mth::SQ(d));
    
    cf[0] = yd - cf[1] * d - cf[2] * mth::SQ(d);
    
    return cf;
  }
  
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
    // ************************************************************** //

  template<typename U, typename T> inline
  void Hunt(U const n, const T* const array, T const& value, U &ilow)
  {
    bool const ascend = (array[n-1] > array[0]) ? true : false;
    U ihigh, index, increment;

    if ((ilow <= U(0))  ||  (ilow > n-1)) {
      
      /* --- Input guess not useful here, go to bisection --  --------- */
      
      ilow = 0;
      ihigh = n;
   
    }else{
      
      /* --- Else hunt up or down to bracket value --    -------------- */ 
      
      increment = 1;
      if (((value >= array[ilow]) ? true : false) == ascend) {
	ihigh = ilow + increment;
	if (ilow == n-1) return;
	
	/* --- Hunt up --                                -------------- */
	
	while (((value >= array[ihigh]) ? true : false) == ascend) {
	  ilow = ihigh;
	  increment += increment;
	  ihigh = ilow + increment;
	  if (ihigh >= n) { ihigh = n;  break; }
	}
      } else {
	ihigh = ilow;
	if (ilow == 0) return;
	
	/* --- Hunt down --                              -------------- */
	
	while (((value <= array[ilow]) ? true : false) == ascend) {
	  ihigh = ilow;
	  increment += increment;
	  ilow = ihigh - increment;
	  if (ilow <= 0) { ilow = 0;  break; }
	}
      }
    }
    
    /* --- Bisection algorithm --                        -------------- */
    
    if (ascend) {
      while (ihigh - ilow > 1) {
	index = (ihigh + ilow) >> 1;
	if (value >= array[index])
	  ilow = index;
	else
	  ihigh = index;
      }
    } else {
      while (ihigh - ilow > 1) {
	index = (ihigh + ilow) >> 1;
	if (value <= array[index])
	  ilow = index;
	else
	  ihigh = index;
      }
    }
  }
  
  // ************************************************************** //

  template<typename U, typename T> inline
  void Locate(U const n,  const T* const array, T value, U &ilow)
  {
    U ihigh = n, index;
    
    bool const ascend = (array[n-1] > array[0]) ? true : false;
    ilow = 0;
    
    if (ascend) {
      while (ihigh - ilow > 1) {
	index = (ihigh + ilow) >> 1;
	if (value >= array[index])
	  ilow = index;
	else
	  ihigh = index;
      }
    } else {
      while (ihigh - ilow > 1) {
	index = (ihigh + ilow) >> 1;
	if (value <= array[index])
	  ilow = index;
	else
	  ihigh = index;
      }
    }
  }

  // ********************************************************************* //

  template<typename U, typename T>
  void interpolation_Linear(U const Ntable,  const T* const __restrict__ xtable,
			    const T* const __restrict__ ytable, U const N,
			    const T* const __restrict__ x, T* const __restrict__ y,
			    bool const hunt = true)
  {
    
    // ---- Hunt / Locate implementation based on NR and RH2001 --- //
    
    bool const ascend = (xtable[1] > xtable[0]) ? true : false;
    T const xmin = (ascend) ? xtable[0] : xtable[Ntable-1];
    T const xmax = (ascend) ? xtable[Ntable-1] : xtable[0];
    U j = 0;

    
    // --- Perform interpolation --- //

    for (int n = 0;  n < N;  n++) {
      if (x[n] <= xmin)
	y[n] = (ascend) ? ytable[0] : ytable[Ntable-1];
      else if (x[n] >= xmax)
	y[n] = (ascend) ? ytable[Ntable-1] : ytable[0];
      else {
	
	// --- Reuse the index from previous interpolated element to speed up
	//     bracketing of the interval. Speeds up A LOT the code!
	
	if (hunt) 
	  Hunt<U,T>(Ntable, xtable, x[n], j);
	else
	  Locate<U,T>(Ntable, xtable, x[n], j);

	// --- weighted average between two points of the interval --- //
	
	T const cint = (xtable[j+1] - x[n]) / (xtable[j+1] - xtable[j]);
	y[n] = cint*ytable[j] + (T(1) - cint)*ytable[j+1];
      }
    }
  }
  // ********************************************************************* //

  template<typename T>
  T signFortran2(const T val)
  {
    return ((val >= T(0))? T(1) : T(-1));
  }
  
  // ********************************************************************* //

  template<typename T>
  T cent_deriv_steffen(T const odx,T const dx, T const yu, T const y0, T const yd)
  {
    /* --- Derivatives from Steffen (1990) --- */
    
    const T S0 = (yd - y0) / dx;
    const T Su = (y0 - yu) / odx;
    const T P0 = std::abs((Su*dx + S0*odx) / (odx+dx)) * 0.5;
    return (signFortran2(S0) + signFortran2(Su)) * std::min(std::abs(Su),std::min(std::abs(S0), P0));
  }
  
  // ********************************************************************* //

  template<typename T> inline
  void interpolation_Hermite(int const N,  const T* const __restrict__ x,  const T* const __restrict__ y,
			     int const N1, const T* const __restrict__ x1, T* const __restrict__ y1)
  {
    // --- Coded by J. de la Cruz Rodriguez (ISP-SU, 2024) --- //
    
    int dn = 1, n0 = 0, n1 = N-1;
    int dj = 1, j0 = 0, j1 = N1-1;
    
    if((x[1]-x[0]) < 0){
      dn = -1, n0 = N-1, n1 = 0;
    }
    
    if((x1[1]-x1[0]) < 0){
      dj = -1, j0 = N1-1, j1 = 0;
    }
    
    
    // --- first calculate derivatives --- //
    
    T* const __restrict__ yp = new T[N]();
    T odx = 0, dx = 0;
    
    
    yp[0] = (y[n0+dn]-y[n0]) / (x[n0+dn]-x[n0]);
    yp[n1] = (y[n1-dn]-y[n1]) / (x[n1-dn]-x[n1]);
    
    for(int n=n0+dn; n != n1; n+=dn){ // avoid both outermost points
      odx = x[n]-x[n-dn];
      dx = x[n+dn]-x[n];
      yp[n] = cent_deriv_steffen(odx,dx,y[n-dn], y[n], y[n+dn]);
    }
    
    
    // --- Now calculate interpolated values --- //

    for(int n=n0; n != n1; n+= dn){
      T const dx = x[n+dn]-x[n];
      T const ypu = yp[n]*dx;
      T const ypc = yp[n+dn]*dx;
      
      for(int j=j0; j != j1+dj; j += dj){
	if((x1[j] <= x[n+dn]) && (x1[j] > x[n])){
	  T const u  = (x1[j]-x[n])/dx;
	  T const u2 = u*u;
	  T const u3 = u2*u;
	  y1[j] = (2.0*u3 - 3.0*u2 + 1.0)*y[n] + (u3-2.0*u2+u)*ypu + (3.0*u2-2.0*u3)*y[n+dn] + (u3-u2)*ypc;
	}
      }
    } // intervals in the real data 
    
    
    
    // --- are there points outside the domain? --- //
    
    T const pmin = y[n0];
    T const pmax = y[n1];
    T const xmin1 = x[n0];
    T const xmax1 = x[n1];
    
    
    for(int j=0; j<N1; ++j){
      if(x1[j] <= xmin1) y1[j] = pmin;
      else if(x1[j] >= xmax1) y1[j] = pmax;
    }
    
    
    delete [] yp;   
  }

  // ********************************************************************* //
  
}


#endif
