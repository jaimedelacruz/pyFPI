#ifndef PREFHPP
#define PREFHPP

/* ---
   Analytical prefilter tools
   Generate a theoretical prefilter curve and its Jacobian

   Pref = pg / (1 + (2*dw/fwhm)**(2*ncav)) * (1 + apodization_value*(p0*dw + p1*dw**2 + p2*dw**3))
   with dw = wav - cw
   
   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
   --- */

#include <cmath>
#include <cstring>
#include <omp.h>

#include "math.hpp"

namespace pref{
  
  template<typename T>
  inline constexpr T const PI_2 = 3.1415926535897932384626433832 * 0.5;
  
  template<typename T>
  inline constexpr T const fwhm_scale = 0.35;

  template<typename T>
  inline constexpr T const fwhm_frac = 1.0 / 5.0;

  template<typename T>
  inline constexpr T const decay = (fwhm_scale<T> + fwhm_frac<T>) / fwhm_frac<T>;

  // *************************************************************** //

  template<typename T>
  inline T apodization(T const twin, T const cw, T const fwhm)
  {
    constexpr T const mid_decay = decay<T>;
    T const decayHalf = fwhm * fwhm_frac<T>;
    T const PI2_decayHalf = PI_2<T> / decayHalf;

    T const tw = twin - cw;
    
    T const tw1 = (tw) * PI2_decayHalf + PI_2<T>*mid_decay;
    T const tw2 = (tw) * PI2_decayHalf - PI_2<T>*mid_decay;
    
    return  T(0.25) * (T(1) + std::tanh(tw1)) * (T(1) - std::tanh(tw2));
  }
  
  // *************************************************************** //

  template<typename T>
  inline T Dapodization(T const twin, T const cw, T const fwhm, T &Da_dcw, T &Da_dfwhm)
  {
    T const decayHalf = fwhm * fwhm_frac<T>;
    T const PI2_decayHalf = PI_2<T> / decayHalf;
    T const dtw_dcw = - PI2_decayHalf;

    T const tw = twin - cw;
    constexpr T const mid_decay = decay<T>;

    T const tw1 = (tw) * PI2_decayHalf + PI_2<T>*mid_decay;
    T const tw2 = (tw) * PI2_decayHalf - PI_2<T>*mid_decay;

    T const dtw1_dfwhm = - PI2_decayHalf * (tw / fwhm);
    T const dtw2_dfwhm = - PI2_decayHalf * (tw / fwhm);

    T const tanh_tw1 =  std::tanh(tw1);
    T const tanh_tw2 =  std::tanh(tw2);

    T const Wapod1 = (T(1) + tanh_tw1);
    T const Wapod2 = (T(1) - tanh_tw2);
    T const dWapod1 = (T(1) - mth::SQ(tanh_tw1));
    T const dWapod2 =  mth::SQ(tanh_tw2) - T(1);
    
    T const Wapod = T(0.25) * Wapod1 * Wapod2;
    
    T const tmp1 = (Wapod2 * dWapod1);
    T const tmp2 = (Wapod1 * dWapod2);
    
    
    Da_dcw = T(0.25) * dtw_dcw * (tmp1+tmp2);
    Da_dfwhm = T(0.25) * (dtw1_dfwhm*tmp1-dtw2_dfwhm*tmp2)*-mth::SignFortran(tw);
    
    return Wapod;
  }
  // *************************************************************** //

  template<typename T>
  inline T prefilter(T const wav, T const pg, T const cw, T const fwhm, \
		     T const ncav, T const p0, T const p1, T const p2, \
		     T const backgr)
  {
    T const apod = apodization(wav,cw,fwhm);
    
    T const dw = wav - cw;
    T const q = T(2) * dw / fwhm;
    T const lin = pg*(T(1) + apod*dw*(p0 + dw*(p1 + p2*dw)));
    T const pref = backgr * lin / (T(1) + std::pow(q*q,ncav));
    
    return pref;
  }

  // *************************************************************** //
  
  template<typename T>
  inline T Dprefilter(T const wav, T const pg, T const cw, T const fwhm, \
		      T const ncav, T const p0, T const p1, T const p2, \
		      T const backgr, T* const dp)
  {
    T dapod_dcw = 0;
    T dapod_dfwhm = 0;
    T const apod = Dapodization(wav,cw,fwhm, dapod_dcw, dapod_dfwhm);
      
    T const dw = wav - cw;
    T const q = T(2) * dw / fwhm;
    T const qa = std::abs(q);
    T const q2n = std::pow(qa,T(2)*ncav);

    T const pref = T(1) / (T(1) + q2n);
    T const pol = dw*(p0 + dw*(p1 + p2*dw));
    T const lin = (T(1) + apod*pol);
    T res =  pg * lin * pref * backgr;

    T const q2n_m2 =  std::pow(qa,T(2)*ncav-T(2));
    
    T const pref2 = pref*pref;
    T const pgpref = pg * pref;
    T const pglin = pg * lin;
    
    T const tmp = pref2 * q2n_m2*q * (T(2)*ncav) / fwhm;
    
    dp[0] = backgr*pref * lin;
    dp[1] = backgr*((T(2) * tmp * pglin - apod*pgpref*(p0 + dw*(T(2)*p1 + T(3)*p2*dw)))  + pgpref*pol*dapod_dcw);
    dp[2] = backgr*(pglin*tmp*q  + pgpref*pol*dapod_dfwhm);
    dp[3] = ((qa == 0)? T(0) : -pref2 * q2n * std::log(qa*qa)) * pglin*backgr;
    
    T const tmp2 = apod*backgr*pgpref * dw;
    dp[4] = tmp2;
    dp[5] = tmp2*dw;
    dp[6] = tmp2*dw*dw;
    
    
    
    return res;
  }

  // *************************************************************** //

  
}

#endif
