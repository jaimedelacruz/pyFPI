/* ---

   FPI class to generate a dual etalon transmission profile
   and its derivatives relative to the cavity and reflectivity
   errors. There are three types of routines, based on name termination:
   
   -  ray: evaluates the profile for a single ray at a given angle.
   - conv: fast approximation that accounts for the slightly converging beam
           at F#165 (for CRISP). It assumes a symmetric beam. Based on
	   Scharmer's ANA routines, with optimizations for speed.
   - full: more accurate calculation, including the tilt of the LRE.
           The angle selection is based on sampling the pupil and performing
	   a histogram of the angular values. This calculation is only
	   done once (and stored for subsequent calculations). Based on Scharmer's
	   ANA routines.
           
   The "individual"-named function return the profiles of the LRE and HRE in
   separate output arrays, instead of returning the product of the two. These
   are used in the LRE-scan dat fitting.

   The class methods are implemented in fpi.cpp and fpi_individual.cpp.

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)

   
   References:
       Scharmer (2006);
       de la Cruz Rodriguez (2010) (numerical project at SU);
       Scharmer, de la Cruz Rodriguez et al. (2013);
       

   Comments:
       The derivatives can be trivially obtained by deriving each equation
       and propagating them with the chain rule. They are nearly identical to
       the finite difference ones, but hopefully faster to compute.
       
   --- */
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <complex>
#include <format>
#include <iostream>

#include "fpi.hpp"
#include "fpi_helper.hpp"
#include "math.hpp"

// ********************************************************************* //

void fpi::FPI::optimize_Zernike()
{
  constexpr const int N = 201;
  constexpr const int Ndef = 25;
  constexpr const ft max_defocus_mm = 6;

  
  // --- conversion from mm --- //
  
  ft const defocus_scl = -PI / (8.0*std::sqrt(3.0)) * mth::SQ(1.0/this->FR) / (this->cw*1.e-8);
  
  std::vector<ft> angles(NRAYS_HR,0.0);
  std::vector<ft> defocus(Ndef,0.0);
  std::vector<ft> pmax(Ndef,0.0);
  
  for(int ii=0; ii<Ndef; ++ii){
    defocus[ii] = ft(ii) / ft(Ndef-1) * max_defocus_mm * defocus_scl; 
  }

  // --- init angles --- //

  for(int ii=0; ii<NRAYS_HR; ++ii){
    angles[ii] = std::acos(this->betah_hr[ii] / (2*PI))*2*this->FR;
  }

  
  // --- create a wavelength array --- //

  std::vector<ft> tw(N), tr(N);
  for(long ii=0; ii<N; ++ii)
    tw[ii] = (ii-N/2)*0.003;


  // --- for each defocus length, tests max peak transmission --- //

  int itmax = 0;
  ft tmax = 0;
      
  for(int id = 0; id<Ndef; ++id){

    // --- fill in zern4 terms --- //

    for(int ii=0; ii<NRAYS_HR; ++ii)
      this->zern4[ii] = std::exp(std::complex<ft>(0.0, -defocus[id]*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0)));

    
    // --- calculate profile --- //

    dual_fpi_full_complex(N,tw.data(), tr.data(), 0.0,0.0,0.0,0.0,false);

    ft ipmax = tr[0];
    
    for(int ii=1; ii<N;++ii){
      ipmax = std::max(ipmax, tr[ii]);
    }

    pmax[id] = ipmax;
    
    if(pmax[id] > tmax){
      tmax = pmax[id];
      itmax = id;
    }
  }

  itmax = std::max(itmax-1,0);


  // --- now bracket the optimal peak using a parabola fit --- //

  std::array<ft,3> c = mth::parab_fit<ft>(&defocus[itmax], &pmax[itmax]);

  ft const defocus_final = - 0.5 * c[1] / c[2];

  

  // --- populate the final zern4 --- //

  for(int ii=0; ii<NRAYS_HR; ++ii){
    ft const zernike_defocus = defocus_final*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0);
    this->zern4[ii] = std::exp(std::complex<ft>(0.0, -zernike_defocus));
  }

}

// ********************************************************************* //

void fpi::FPI::optimize_Zernike_conv()
{
  constexpr const int N = 201;
  constexpr const int Ndef = 25;
  constexpr const ft max_defocus_mm = 6;

  
  // --- conversion from mm --- //
  
  ft const defocus_scl = -PI / (8.0*std::sqrt(3.0)) * mth::SQ(1.0/this->FR) / (this->cw*1.e-8);
  
  std::vector<ft> angles(fpi::NRAYS,0.0);
  std::vector<ft> defocus(Ndef,0.0);
  std::vector<ft> pmax(Ndef,0.0);
  
  for(int ii=0; ii<Ndef; ++ii){
    defocus[ii] = ft(ii) / ft(Ndef-1) * max_defocus_mm * defocus_scl; 
  }

  // --- init angles --- //

  for(int ii=0; ii<fpi::NRAYS; ++ii){
    angles[ii] = std::acos(this->calp[ii] / (2*PI))*2*this->FR;
  }

  
  // --- create a wavelength array --- //

  std::vector<ft> tw(N), tr(N);
  for(long ii=0; ii<N; ++ii)
    tw[ii] = (ii-N/2)*0.003;


  // --- for each defocus length, tests max peak transmission --- //

  int itmax = 0;
  ft tmax = 0;
      
  for(int id = 0; id<Ndef; ++id){

    // --- fill in zern4 terms --- //

    for(int ii=0; ii<fpi::NRAYS; ++ii)
      this->zern4_conv[ii] = std::exp(std::complex<ft>(0.0, -defocus[id]*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0)));

    
    // --- calculate profile --- //

    dual_fpi_conv_complex(N,tw.data(), tr.data(), 0.0,0.0,0.0,0.0,false);

    ft ipmax = tr[0];
    
    for(int ii=1; ii<N;++ii){
      ipmax = std::max(ipmax, tr[ii]);
    }

    pmax[id] = ipmax;
    
    if(pmax[id] > tmax){
      tmax = pmax[id];
      itmax = id;
    }
  }

  itmax = std::max(itmax-1,0);


  // --- now bracket the optimal peak using a parabola fit --- //

  std::array<ft,3> c = mth::parab_fit<ft>(&defocus[itmax], &pmax[itmax]);

  ft const defocus_final = - 0.5 * c[1] / c[2];

  

  // --- populate the final zern4 --- //

  for(int ii=0; ii<fpi::NRAYS; ++ii){
    ft const zernike_defocus = defocus_final*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0);
    this->zern4_conv[ii] = std::exp(std::complex<ft>(0.0, -zernike_defocus));
  }

}

// ********************************************************************* //

void fpi::FPI::dual_fpi_full_complex(int const N1, const ft* const tw, ft* const tr,
				     ft const erh, ft const erl, ft const ech,
				     ft const ecl, bool const normalize)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);

  
  // --- Get sinp, note that psi is in fact psi/2 --- //
  
  Arr2D<ft> psi_lr(NRAYS_LR, N1);
  Arr2D<ft> psi_hr(NRAYS_HR, N1);
    
  fpi::Arr2D<ft> sinp_hr = fpi::get_psi2(N1,cw+BlueShift,tw,hc+ech,betah_hr, psi_hr);
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  fpi::Arr2D<ft> sinp_lr = fpi::get_psi2(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr, psi_lr);

  
  // --- construct the electric field transmission profile with the angle integral --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  std::vector<std::complex<ft>> tr_nu(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr(N1);
  std::vector<std::complex<ft>> tr_hr(N1);
  
  for(int n=0; n<NRAYS_LR; ++n){
    
    for(int ww=0; ww<N1; ++ww){
      ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr(n,ww)));
      ft const cosp_lr = std::cos(psi_lr(n,ww));
      tr_lr[ww] = std::complex<ft>(lre_real*(ft(1)-tlr)*cosp_lr, lre_real*(ft(1)+tlr)*sinp_lr(n,ww));
    }

    
    for(int m=0; m<NRAYS_HR; ++m){
      
      if(n_betah(n,m) > 1.e-5){
	
	for(int ww=0; ww<N1; ++ww){  	 
	  ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr(m,ww)));
	  ft const cosp_hr = std::cos(psi_hr(m,ww));
	  tr_hr[ww] = std::complex<ft>(hre_real*(ft(1)-thr)*cosp_hr, hre_real*(ft(1)+thr)*sinp_hr(m,ww));
	}

	std::complex<ft> const zern4_Wnm = this->zern4[m] * this->n_betah(n,m);

	for(int ww=0; ww<N1; ++ww){  	  

	  // --- now multiply the two electric field transmission profiles with the focus term --- //
	  
	  tr_nu[ww] += tr_hr[ww] * tr_lr[ww] * zern4_Wnm;
	  
	} // ww
      } // if
    } // m
  } // n

  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    tr[ww] = (tr_nu[ww]*std::conj(tr_nu[ww])).real();
  }  

  
  // --- Area normalization --- //
  
  if(normalize){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += tr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) tr[ii] *= suma;
  }

}
  
// ********************************************************************* //

void CosMany(long const N, ft* const __restrict__ d)
{
  for(long ii=0; ii<N; ++ii){
    d[ii] = std::cos(d[ii]);
  }   
}

// ********************************************************************* //

void SinMany(long const N, ft* const __restrict__ d)
{
  for(long ii=0; ii<N; ++ii){
    d[ii] = std::sin(d[ii]);
  }   
}

// ********************************************************************* //

template<typename T>
inline void sincos(T const& psi, T* const sinpsi, T* const cospsi)
{
  *sinpsi = std::sin(psi);
  *cospsi = std::cos(psi);
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_full_complex_der(int const N1, const ft* const tw, ft* const tr,
					 ft* const dtr, ft const erh, ft const erl, ft const ech,
					 ft const ecl, bool const normalize)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  ft const dfhr_derh = 4*(ft(1)+thr) / mth::CUB(ft(1) - thr);
  ft const dflr_derl = 4*(ft(1)+tlr) / mth::CUB(ft(1) - tlr);
  
  
  // --- Get sinp, note that psi is in fact psi/2 --- //
  
  Arr2D<ft> psi_lr(NRAYS_LR, N1), dpsi_lr(NRAYS_LR, N1);
  Arr2D<ft> psi_hr(NRAYS_HR, N1), dpsi_hr(NRAYS_HR, N1);
  
  fpi::Arr2D<ft> sinp_hr = fpi::get_psi2_der(N1,cw+BlueShift,tw,hc+ech,betah_hr, psi_hr, dpsi_hr);
  Arr2D<ft> cosp_hr = psi_hr;
  CosMany(NRAYS_HR*N1, &cosp_hr(0,0));
  
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  ft const decl_ech = lc/hc;

  fpi::Arr2D<ft> sinp_lr = fpi::get_psi2_der(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr, psi_lr, dpsi_lr);
  Arr2D<ft> cosp_lr = psi_lr;
  CosMany(NRAYS_LR*N1, &cosp_lr(0,0));


  // --- Init temporary arrays --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  std::vector<std::complex<ft>> tr_nu(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr(N1);
  
  std::vector<std::complex<ft>> dtr_dtlr(N1, zero_complex);
  std::vector<std::complex<ft>> dtr_dthr(N1, zero_complex);

  std::vector<std::complex<ft>> dtr_dcl(N1, zero_complex);
  std::vector<std::complex<ft>> dtr_dch(N1, zero_complex);
  

  // --- construct the electric field transmission profile with the angle integral --- //

  for(int n=0; n<NRAYS_LR; ++n){


    // --- precompute the LRE profile (this part can be vectorized) --- //
    
    for(int ww=0; ww<N1; ++ww){
      
      ft const lre_real = cLRE / ( ft(1) + flr * mth::SQ(sinp_lr(n,ww)));
      tr_lr[ww] = std::complex<ft>(lre_real *(ft(1)-tlr)*cosp_lr(n,ww),lre_real *(ft(1)+tlr)*sinp_lr(n,ww));
    }

    
    for(int m=0; m<NRAYS_HR; ++m){
      
      if(n_betah(n,m) > 1.e-5){
	
	std::complex<ft> const zern4_Wnm = this->zern4[m] * this->n_betah(n,m);

	for(int ww=0; ww<N1; ++ww){
	  
	  ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr(n,ww));     
	  ft const lre_real = cLRE / denom_lr;

	  
	  ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr(m,ww));
	  ft const hre_real = cHRE / denom_hr;

	  
	  // --- HR profile --- //
	  
	  std::complex<ft> const tr_hr = std::complex<ft>(hre_real * (ft(1)-thr)*cosp_hr(m,ww), hre_real *(ft(1)+thr)*sinp_hr(m,ww));


	  
	  // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
	  
	  std::complex<ft> dlr_dcl = dpsi_lr(n,ww)*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr(n,ww),(ft(1)+tlr)*cosp_lr(n,ww)) -
						    (ft(2)*flr*sinp_lr(n,ww)*cosp_lr(n,ww)/denom_lr) * tr_lr[ww]);
	  

	  
	  // --- derivative with respect to CL --- //
	  
	  dtr_dcl[ww] += dlr_dcl * tr_hr * zern4_Wnm;


	  
	  // --- derivative with respect to CH --- //
	  
	  dtr_dch[ww] += zern4_Wnm * ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr(m,ww),(ft(1)+thr)*cosp_hr(m,ww)) -
				       (2*fhr*sinp_hr(m,ww)*cosp_hr(m,ww)/denom_hr) * tr_hr)*dpsi_hr(m,ww) * tr_lr[ww] + dlr_dcl * decl_ech * tr_hr);


	  
	  // --- derivative with respect to HR --- //
      
	  dtr_dthr[ww] += zern4_Wnm * tr_lr[ww] * (tr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr(m,ww))/denom_hr)) +
						   hre_real * (std::complex<ft>(-cosp_hr(m,ww),sinp_hr(m,ww))));
      

	  	  
	  // --- derivative with respect to LR --- //
	  
	  dtr_dtlr[ww] += zern4_Wnm * tr_hr * (tr_lr[ww] * (cLRE - (dflr_derl * mth::SQ(sinp_lr(n,ww))/denom_lr)) +
					       lre_real * (std::complex<ft>(-cosp_lr(n,ww),sinp_lr(n,ww))));
	  

	  // --- now multiply the two transmission profiles --- //
	  
	  tr_nu[ww] += tr_hr * tr_lr[ww] * zern4_Wnm;
	  
	}
	
      } // if integration weight is not zero 
    } // m
  } // n


  // --- Init pointers for derivatives --- //
  
  ft* const __restrict__ dtr_derh = dtr;
  ft* const __restrict__ dtr_derl = dtr + 1*N1;
  ft* const __restrict__ dtr_dech = dtr + 2*N1;
  ft* const __restrict__ dtr_decl = dtr + 3*N1;
  
  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    tr[ww] = (tr_nu[ww]*std::conj(tr_nu[ww])).real();

    // --- apply the chain rule to the derivative of tr --- //
    
    dtr_derl[ww] = (tr_nu[ww]*std::conj(dtr_dtlr[ww]) + std::conj(tr_nu[ww])*dtr_dtlr[ww]).real(); 
    dtr_derh[ww] = (tr_nu[ww]*std::conj(dtr_dthr[ww]) + std::conj(tr_nu[ww])*dtr_dthr[ww]).real();
    dtr_decl[ww] = (tr_nu[ww]*std::conj(dtr_dcl[ww])  + std::conj(tr_nu[ww])*dtr_dcl[ww]).real();
    dtr_dech[ww] = (tr_nu[ww]*std::conj(dtr_dch[ww])  + std::conj(tr_nu[ww])*dtr_dch[ww]).real();
  }
  
  
  // --- Area normalization, apply the chain rule to the derivative of the reflectivity --- //

  if(normalize){
    ft sum = ft(0);
    ft sum1 = ft(0);
    ft sum2 = ft(0);
    
    
    for(int ii=0; ii<N1; ++ii){
      sum += tr[ii];
      sum1+= dtr_derh[ii];
      sum2+= dtr_derl[ii];
    }
    sum = ft(1) / sum;
    ft const sum3 = sum*sum;
    
    for(int ii=0; ii<N1; ++ii){
      
      dtr_dech[ii] *= sum;
      dtr_decl[ii] *= sum;
      dtr_derh[ii] = dtr_derh[ii]*sum - sum3*tr[ii]*sum1;	
      dtr_derl[ii] = dtr_derl[ii]*sum - sum3*tr[ii]*sum2;
      tr[ii] *= sum;
    }
  }

}
  
// ********************************************************************* //

void fpi::FPI::dual_fpi_full_complex_individual(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
					        ft const erh, ft const erl, ft const ech,
						ft const ecl,  bool const normalize_ltr,
						bool const normalize_htr)const
{

  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);

  
  // --- Get sinp, note that psi is in fact psi/2 --- //
  
  Arr2D<ft> psi_lr(NRAYS_LR, N1);
  Arr2D<ft> psi_hr(NRAYS_HR, N1);
    
  fpi::Arr2D<ft> sinp_hr = fpi::get_psi2(N1,cw+BlueShift,tw,hc+ech,betah_hr, psi_hr);
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  fpi::Arr2D<ft> sinp_lr = fpi::get_psi2(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr, psi_lr);

  
  
  // --- construct the electric field transmission profile with the angle integral --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);
  
  std::vector<std::complex<ft>> tr_lr(N1, zero_complex);
  std::vector<std::complex<ft>> tr_hr(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr_tmp(N1);

  for(int n=0; n<NRAYS_LR; ++n){
    
    for(int ww=0; ww<N1; ++ww){
      ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr(n,ww)));
      ft const cosp_lr = std::cos(psi_lr(n,ww));
      tr_lr_tmp[ww] = std::complex<ft>(lre_real*(ft(1)-tlr)*cosp_lr, lre_real*(ft(1)+tlr)*sinp_lr(n,ww));
    }
      
    
    for(int m=0; m<NRAYS_HR; ++m){
      
      if(n_betah(n,m) > 1.e-5){
	
	std::complex<ft> const zern4_Wnm = this->zern4[m] * this->n_betah(n,m);

	for(int ww=0; ww<N1; ++ww){  	 
	  ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr(m,ww)));
	  ft const cosp_hr = std::cos(psi_hr(m,ww));
	  tr_hr[ww] += zern4_Wnm * std::complex<ft>(hre_real*(ft(1)-thr)*cosp_hr, hre_real*(ft(1)+thr)*sinp_hr(m,ww));
	  tr_lr[ww] += zern4_Wnm * tr_lr_tmp[ww];
	}
	
      } // if
    } // m
  } // n

  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    htr[ww] = (tr_hr[ww]*std::conj(tr_hr[ww])).real();
    ltr[ww] = (tr_lr[ww]*std::conj(tr_lr[ww])).real();
  }  
  

  // --- Area normalization LRE --- //

  if(normalize_ltr){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += ltr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) ltr[ii] *= suma;
  }


  // --- Area normalization HRE --- //
  
  if(normalize_htr){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += htr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) htr[ii] *= suma;
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_full_complex_individual_der(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
						    ft* const dtr, ft const erh, ft const erl, ft const ech,
						    ft const ecl,  bool const normalize_ltr,
						    bool const normalize_htr)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  ft const dfhr_derh = 4*(ft(1)+thr) / mth::CUB(ft(1) - thr);
  ft const dflr_derl = 4*(ft(1)+tlr) / mth::CUB(ft(1) - tlr);
  
  
  // --- Get sinp, note that psi is in fact psi/2 --- //
  
  Arr2D<ft> psi_lr(NRAYS_LR, N1), dpsi_lr(NRAYS_LR, N1);
  Arr2D<ft> psi_hr(NRAYS_HR, N1), dpsi_hr(NRAYS_HR, N1);
  
  fpi::Arr2D<ft> sinp_hr = fpi::get_psi2_der(N1,cw+BlueShift,tw,hc+ech,betah_hr, psi_hr, dpsi_hr);
  Arr2D<ft> cosp_hr = psi_hr;
  CosMany(NRAYS_HR*N1, &cosp_hr(0,0));
  
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  ft const decl_ech = lc/hc;

  fpi::Arr2D<ft> sinp_lr = fpi::get_psi2_der(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr, psi_lr, dpsi_lr);
  Arr2D<ft> cosp_lr = psi_lr;
  CosMany(NRAYS_LR*N1, &cosp_lr(0,0));


  // --- Init temporary arrays --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  std::vector<std::complex<ft>> tr_hr(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr_tmp(N1);
  
  std::vector<std::complex<ft>> dlr_dtlr(N1, zero_complex);
  std::vector<std::complex<ft>> dhr_dthr(N1, zero_complex);

  std::vector<std::complex<ft>> dlr_dcl(N1, zero_complex);
  std::vector<std::complex<ft>> dhr_dch(N1, zero_complex);
  std::vector<std::complex<ft>> dlr_dch(N1, zero_complex); // dTR_lr / dch
  

  // --- construct the electric field transmission profile with the angle integral --- //

  for(int n=0; n<NRAYS_LR; ++n){


    // --- precompute the LRE profile (this part can be vectorized) --- //
    
    for(int ww=0; ww<N1; ++ww){
      ft const lre_real = cLRE / ( ft(1) + flr * mth::SQ(sinp_lr(n,ww)));
      tr_lr_tmp[ww] = std::complex<ft>(lre_real *(ft(1)-tlr)*cosp_lr(n,ww),lre_real *(ft(1)+tlr)*sinp_lr(n,ww));
    }

    
    for(int m=0; m<NRAYS_HR; ++m){
      
      if(n_betah(n,m) > 1.e-5){
	
	std::complex<ft> const zern4_Wnm = this->zern4[m] * this->n_betah(n,m);

	for(int ww=0; ww<N1; ++ww){
	  
	  ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr(n,ww));     
	  ft const lre_real = cLRE / denom_lr;

	  
	  ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr(m,ww));
	  ft const hre_real = cHRE / denom_hr;

	  
	  // --- HR profile --- //
	  
	  std::complex<ft> const itr_hr = std::complex<ft>(hre_real * (ft(1)-thr)*cosp_hr(m,ww), hre_real *(ft(1)+thr)*sinp_hr(m,ww));


	  
	  // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
	  
	  std::complex<ft> idlr_dcl = dpsi_lr(n,ww)*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr(n,ww),(ft(1)+tlr)*cosp_lr(n,ww)) -
						    (ft(2)*flr*sinp_lr(n,ww)*cosp_lr(n,ww)/denom_lr) * tr_lr_tmp[ww])* zern4_Wnm;
	  

	  
	  // --- derivative with respect to CL --- //
	  
	  dlr_dcl[ww] += idlr_dcl; // integration weight is already included


	  
	  // --- derivative with respect to CH --- //
	  
	  dhr_dch[ww] += zern4_Wnm * ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr(m,ww),(ft(1)+thr)*cosp_hr(m,ww)) -
				       (2*fhr*sinp_hr(m,ww)*cosp_hr(m,ww)/denom_hr) * itr_hr)*dpsi_hr(m,ww));
	  
	  dlr_dch[ww] += idlr_dcl * decl_ech; // integration weight is already included
	    

	  
	  // --- derivative with respect to HR --- //
      
	  dhr_dthr[ww] += zern4_Wnm * (itr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr(m,ww))/denom_hr)) +
				       hre_real * (std::complex<ft>(-cosp_hr(m,ww),sinp_hr(m,ww))));
	  

	  	  
	  // --- derivative with respect to LR --- //
	  
	  dlr_dtlr[ww] += zern4_Wnm * (tr_lr_tmp[ww] * (cLRE - (dflr_derl * mth::SQ(sinp_lr(n,ww))/denom_lr)) +
				       lre_real * (std::complex<ft>(-cosp_lr(n,ww),sinp_lr(n,ww))));
	  

	  tr_lr[ww] += zern4_Wnm * tr_lr_tmp[ww];
	  tr_hr[ww] += zern4_Wnm * itr_hr;
	  
	}
	
      } // if integration weight is not zero 
    } // m
  } // n

  
  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;
  ft* const dltr_dech = dtr + 4*N1;
  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    htr[ww] = (tr_hr[ww]*std::conj(tr_hr[ww])).real();
    ltr[ww] = (tr_lr[ww]*std::conj(tr_lr[ww])).real();

    
    // --- apply the chain rule to the derivative of tr --- //
    
    dtr_derl[ww] = (tr_lr[ww]*std::conj(dlr_dtlr[ww]) + std::conj(tr_lr[ww])*dlr_dtlr[ww]).real();
    dtr_derh[ww] = (tr_hr[ww]*std::conj(dhr_dthr[ww]) + std::conj(tr_hr[ww])*dhr_dthr[ww]).real();
    dtr_decl[ww] = (tr_lr[ww]*std::conj(dlr_dcl[ww])  + std::conj(tr_lr[ww])*dlr_dcl[ww]).real();
    dtr_dech[ww] = (tr_hr[ww]*std::conj(dhr_dch[ww])  + std::conj(tr_hr[ww])*dhr_dch[ww]).real();
    dltr_dech[ww] = (tr_lr[ww]*std::conj(dlr_dch[ww])  + std::conj(tr_lr[ww])*dlr_dch[ww]).real();
  }
  
  
  // --- Area normalization LRE --- //
  
  if(normalize_ltr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += ltr[ii];
      sum1+= dtr_derl[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_decl[ii]  *= sum;
      dltr_dech[ii] *= sum;
      dtr_derl[ii] = dtr_derl[ii]*sum - sum2*ltr[ii];
      ltr[ii] *= sum; 
    }
  }
  
  // --- Area normalization HRE --- //

  if(normalize_htr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += htr[ii];
      sum1+= dtr_derh[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_dech[ii] *= sum;
      dtr_derh[ii]  = dtr_derh[ii]*sum - sum2*htr[ii];
      htr[ii]      *= sum; 
    }
  }

}

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_complex(int const N1, const ft* const tw, ft* const tr,
				     ft const erh, ft const erl, ft const ech,
				     ft const ecl, bool const normalize)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);

  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error

  
  
  // --- construct the electric field transmission profile with the angle integral --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  std::vector<std::complex<ft>> tr_nu(N1, zero_complex);

  std::vector<ft> phi_lr(N1), phi_hr(N1);
  std::vector<ft> sinp_lr(N1), sinp_hr(N1);
  std::vector<ft> cosp_lr(N1), cosp_hr(N1);


  
  // --- Perform angular integral --- //
  
  for(int n = 0; n < fpi::NRAYS; ++n){

    ft const plr = calp[n]*(lc+ecl_ech);
    ft const phr = calp[n]*(hc+ech);

    
    // --- precalculate the phases --- //
    
    for(int ww=0; ww<N1; ++ww){
      ft const wav1 = tw[ww] + cw + BlueShift;
      sinp_lr[ww] = cosp_lr[ww] = phi_lr[ww] = plr / wav1;
      sinp_hr[ww] = cosp_hr[ww] = phi_hr[ww] = phr / wav1;
    }

    
    // --- calculate (vectorized) sines and cosines of the phase --- //
    
    SinMany(N1, sinp_lr.data());
    SinMany(N1, sinp_hr.data());
    CosMany(N1, cosp_lr.data());
    CosMany(N1, cosp_hr.data());
    

    
    // --- calculate the electric field transmission profiles with the refocussing term --- //

    std::complex<ft> const zern4_Wnm = this->zern4_conv[n] * this->wng[n];

    for(int ww=0; ww<N1; ++ww){
      ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr[ww]));
      std::complex<ft> tr_lr(lre_real*(ft(1)-tlr)*cosp_lr[ww], lre_real*(ft(1)+tlr)*sinp_lr[ww]);

      ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr[ww]));
      std::complex<ft> tr_hr(hre_real*(ft(1)-thr)*cosp_hr[ww], hre_real*(ft(1)+thr)*sinp_hr[ww]);
      
      tr_nu[ww] += tr_lr*tr_hr*zern4_Wnm;
    } // ww
  } // n

  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    tr[ww] = (tr_nu[ww]*std::conj(tr_nu[ww])).real();
  }  


  
  // --- Area normalization --- //
  
  if(normalize){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += tr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) tr[ii] *= suma;
  }

}

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_complex_der(int const N1, const ft* const tw, ft* const tr,
					 ft* const dtr, ft const erh, ft const erl, ft const ech,
					 ft const ecl, bool const normalize)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  ft const dfhr_derh = 4*(ft(1)+thr) / mth::CUB(ft(1) - thr);
  ft const dflr_derl = 4*(ft(1)+tlr) / mth::CUB(ft(1) - tlr);
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  
  // --- Get sinp, note that psi is in fact psi/2 --- //

  std::vector<ft> phi_lr(N1), phi_hr(N1), dpsi_lr(N1), dpsi_hr(N1);
  std::vector<ft> sinp_lr(N1), sinp_hr(N1);
  std::vector<ft> cosp_lr(N1), cosp_hr(N1);

  
  // --- Init temporary arrays --- //
  
  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);
  
  std::vector<std::complex<ft>> tr_nu(N1, zero_complex);
  std::vector<std::complex<ft>> tr_lr(N1);
  
  std::vector<std::complex<ft>> dtr_dtlr(N1, zero_complex);
  std::vector<std::complex<ft>> dtr_dthr(N1, zero_complex);

  std::vector<std::complex<ft>> dtr_dcl(N1, zero_complex);
  std::vector<std::complex<ft>> dtr_dch(N1, zero_complex);
  

  // --- construct the electric field transmission profile with the angle integral --- //

  for(int n=0; n<fpi::NRAYS; ++n){

    ft const plr = calp[n]*(lc+ecl_ech);
    ft const phr = calp[n]*(hc+ech);
    
    ft const dphr_dhc = calp[n];
    ft const dplr_dlc = calp[n];
    
    
    // --- precalculate the phases --- //
    
    for(int ww=0; ww<N1; ++ww){
      ft const wav1 = 1.0 / (tw[ww] + cw + BlueShift);
      sinp_lr[ww] = cosp_lr[ww] = phi_lr[ww] = plr * wav1;
      sinp_hr[ww] = cosp_hr[ww] = phi_hr[ww] = phr * wav1;
      dpsi_lr[ww] = dplr_dlc * wav1;
      dpsi_hr[ww] = dphr_dhc * wav1;
    }

    
    // --- calculate (vectorized) sines and cosines of the phase --- //
    
    SinMany(N1, sinp_lr.data());
    SinMany(N1, sinp_hr.data());
    CosMany(N1, cosp_lr.data());
    CosMany(N1, cosp_hr.data());

    
    // --- precompute the LRE profile (this part can be vectorized) --- //

    std::complex<ft> const zern4_Wnm = this->zern4_conv[n] * this->wng[n];

    
    for(int ww=0; ww<N1; ++ww){
      ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr[ww]);
      ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr[ww]);

      ft const lre_real = cLRE / denom_lr;
      ft const hre_real = cHRE / denom_hr;
      
      std::complex<ft> const tr_lr(lre_real *(ft(1)-tlr)*cosp_lr[ww],lre_real *(ft(1)+tlr)*sinp_lr[ww]);
      std::complex<ft> const tr_hr(hre_real * (ft(1)-thr)*cosp_hr[ww], hre_real *(ft(1)+thr)*sinp_hr[ww]);

	  
      // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
      
      std::complex<ft> const dlr_dcl = dpsi_lr[ww]*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr[ww],(ft(1)+tlr)*cosp_lr[ww]) -
						      (ft(2)*flr*sinp_lr[ww]*cosp_lr[ww]/denom_lr) * tr_lr);
      
      
      
      // --- derivative with respect to CL --- //
      
      dtr_dcl[ww] += dlr_dcl * tr_hr * zern4_Wnm;
      
      
      
      // --- derivative with respect to CH --- //
      
      dtr_dch[ww] += zern4_Wnm * ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr[ww],(ft(1)+thr)*cosp_hr[ww]) -
				   (2*fhr*sinp_hr[ww]*cosp_hr[ww]/denom_hr) * tr_hr)*dpsi_hr[ww] * tr_lr + dlr_dcl * decl_ech * tr_hr);
      
      
      
      // --- derivative with respect to HR --- //
      
      dtr_dthr[ww] += zern4_Wnm * tr_lr * (tr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr[ww])/denom_hr)) +
					   hre_real * (std::complex<ft>(-cosp_hr[ww],sinp_hr[ww])));
      
      
      
      // --- derivative with respect to LR --- //
      
      dtr_dtlr[ww] += zern4_Wnm * tr_hr * (tr_lr * (cLRE - (dflr_derl * mth::SQ(sinp_lr[ww])/denom_lr)) +
					   lre_real * (std::complex<ft>(-cosp_lr[ww],sinp_lr[ww])));

      
      
      // --- now multiply the two transmission profiles --- //
      
      tr_nu[ww] += tr_hr * tr_lr * zern4_Wnm;
      
    }
    
  } // n
  

  // --- Init pointers for derivatives --- //
  
  ft* const __restrict__ dtr_derh = dtr;
  ft* const __restrict__ dtr_derl = dtr + 1*N1;
  ft* const __restrict__ dtr_dech = dtr + 2*N1;
  ft* const __restrict__ dtr_decl = dtr + 3*N1;
  
  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    tr[ww] = (tr_nu[ww]*std::conj(tr_nu[ww])).real();

    // --- apply the chain rule to the derivative of tr --- //
    
    dtr_derl[ww] = (tr_nu[ww]*std::conj(dtr_dtlr[ww]) + std::conj(tr_nu[ww])*dtr_dtlr[ww]).real(); 
    dtr_derh[ww] = (tr_nu[ww]*std::conj(dtr_dthr[ww]) + std::conj(tr_nu[ww])*dtr_dthr[ww]).real();
    dtr_decl[ww] = (tr_nu[ww]*std::conj(dtr_dcl[ww])  + std::conj(tr_nu[ww])*dtr_dcl[ww]).real();
    dtr_dech[ww] = (tr_nu[ww]*std::conj(dtr_dch[ww])  + std::conj(tr_nu[ww])*dtr_dch[ww]).real();
  }
  
  
  // --- Area normalization, apply the chain rule to the derivative of the reflectivity --- //

  if(normalize){
    ft sum = ft(0);
    ft sum1 = ft(0);
    ft sum2 = ft(0);
    
    
    for(int ii=0; ii<N1; ++ii){
      sum += tr[ii];
      sum1+= dtr_derh[ii];
      sum2+= dtr_derl[ii];
    }
    sum = ft(1) / sum;
    ft const sum3 = sum*sum;
    
    for(int ii=0; ii<N1; ++ii){
      
      dtr_dech[ii] *= sum;
      dtr_decl[ii] *= sum;
      dtr_derh[ii] = dtr_derh[ii]*sum - sum3*tr[ii]*sum1;	
      dtr_derl[ii] = dtr_derl[ii]*sum - sum3*tr[ii]*sum2;
      tr[ii] *= sum;
    }
  }

}

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_complex_individual(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
					        ft const erh, ft const erl, ft const ech,
						ft const ecl,  bool const normalize_ltr,
						bool const normalize_htr)const
{
  
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  
  
  // --- construct the electric field transmission profile with the angle integral --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error


  // --- allocate temporary arrays --- //
  
  std::complex<ft>* const __restrict__ tr_lr = new std::complex<ft>[2*N1](); 
  std::complex<ft>* const __restrict__ tr_hr = tr_lr + N1;

  ft* const __restrict__ sinp_lr = new ft[4*N1];
  ft* const __restrict__ sinp_hr = sinp_lr + N1;
  ft* const __restrict__ cosp_lr = sinp_lr + 2*N1;
  ft* const __restrict__ cosp_hr = sinp_lr + 3*N1;
  
  
  for(int n=0; n<fpi::NRAYS; ++n){

    ft const plr = calp[n]*(lc+ecl_ech);
    ft const phr = calp[n]*(hc+ech);
    
    for(int ww=0; ww<N1; ++ww){
      
      ft const wav1 = 1.0 / (tw[ww] + cw + BlueShift);
      sinp_lr[ww] = cosp_lr[ww] = plr * wav1;
      sinp_hr[ww] = cosp_hr[ww] = phr * wav1;
    }

    // --- Calculate sines and cosines with vectorized functions --- //
    
    SinMany(2*N1, sinp_lr); // doing both etalons in one go as they are consecutive in mem.
    CosMany(2*N1, cosp_lr); // doing both etalons in one go as they are consecutive in mem.
    
    std::complex<ft> const zern4_Wnm = this->zern4_conv[n] * this->wng[n];

    for(int ww=0; ww<N1; ++ww){
      ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr[ww]));
      ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr[ww]));
      
      tr_hr[ww] += zern4_Wnm * std::complex<ft>(hre_real*(ft(1)-thr)*cosp_hr[ww], hre_real*(ft(1)+thr)*sinp_hr[ww]);
      tr_lr[ww] += zern4_Wnm * std::complex<ft>(lre_real*(ft(1)-tlr)*cosp_lr[ww], lre_real*(ft(1)+tlr)*sinp_lr[ww]);
    }
  } // n
  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    htr[ww] = (tr_hr[ww]*std::conj(tr_hr[ww])).real();
    ltr[ww] = (tr_lr[ww]*std::conj(tr_lr[ww])).real();
  }  

  
  // --- clean-up --- //
  
  delete [] tr_lr;
  delete [] sinp_lr;
  
  
  // --- Area normalization LRE --- //

  if(normalize_ltr){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += ltr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) ltr[ii] *= suma;
  }


  // --- Area normalization HRE --- //
  
  if(normalize_htr){
    ft suma = ft(0);
    for(int ii=0; ii<N1; ++ii) suma += htr[ii];
    suma = ft(1) / suma;
    for(int ii=0; ii<N1; ++ii) htr[ii] *= suma;
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_complex_individual_der(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
						    ft* const dtr, ft const erh, ft const erl, ft const ech,
						    ft const ecl,  bool const normalize_ltr,
						    bool const normalize_htr)const
{
  constexpr std::complex<ft> const zero_complex(ft(0),ft(0));
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  ft const dfhr_derh = 4*(ft(1)+thr) / mth::CUB(ft(1) - thr);
  ft const dflr_derl = 4*(ft(1)+tlr) / mth::CUB(ft(1) - tlr);
  
  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  ft const decl_ech = lc/hc;

  
  // --- Init temporary arrays --- //

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  std::complex<ft>* const __restrict__ tr_hr = new std::complex<ft>[7*N1]();
  std::complex<ft>* const __restrict__ tr_lr = tr_hr + N1;
 
  std::complex<ft>* const __restrict__ dlr_dtlr = tr_hr + 2*N1;
  std::complex<ft>* const __restrict__ dhr_dthr = tr_hr + 3*N1;
  std::complex<ft>* const __restrict__ dlr_dcl  = tr_hr + 4*N1;
  std::complex<ft>* const __restrict__ dlr_dch  = tr_hr + 5*N1;
  std::complex<ft>* const __restrict__ dhr_dch  = tr_hr + 6*N1;

  ft* const __restrict__ sinp_lr = new ft[8*N1];
  ft* const __restrict__ sinp_hr = sinp_lr + N1;
  ft* const __restrict__ cosp_lr = sinp_lr + 2*N1;
  ft* const __restrict__ cosp_hr = sinp_lr + 3*N1;
  ft* const __restrict__ psi_lr  = sinp_lr + 4*N1;
  ft* const __restrict__ psi_hr  = sinp_lr + 5*N1;
  ft* const __restrict__ dpsi_lr = sinp_lr + 6*N1;
  ft* const __restrict__ dpsi_hr = sinp_lr + 7*N1;
  

  // --- construct the electric field transmission profile with the angle integral --- //

  for(int n=0; n<fpi::NRAYS; ++n){

    ft const plr = calp[n]*(lc+ecl_ech);
    ft const phr = calp[n]*(hc+ech);
    
    ft const dphr_dhc = calp[n];
    ft const dplr_dlc = calp[n];
    
    
    for(int ww=0; ww<N1; ++ww){
      
      ft const wav1 = 1.0 / (tw[ww] + cw + BlueShift);
      sinp_lr[ww] = cosp_lr[ww] = psi_lr[ww] = plr * wav1;
      sinp_hr[ww] = cosp_hr[ww] = psi_hr[ww] = phr * wav1;
      dpsi_lr[ww] = dplr_dlc * wav1;
      dpsi_hr[ww] = dphr_dhc * wav1;
    }

    // --- Calculate sines and cosines with vectorized functions --- //
    
    SinMany(2*N1, sinp_lr); // doing both etalons in one go as they are consecutive in mem.
    CosMany(2*N1, cosp_lr); // doing both etalons in one go as they are consecutive in mem.
    
    std::complex<ft> const zern4_Wnm = this->zern4_conv[n] * this->wng[n];

    for(int ww=0; ww<N1; ++ww){
	  
      ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr[ww]);     
      ft const lre_real = cLRE / denom_lr;
      
      
      ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr[ww]);
      ft const hre_real = cHRE / denom_hr;
      
	  
      // --- HR profile --- //
      
      std::complex<ft> const itr_hr(hre_real * (ft(1)-thr)*cosp_hr[ww], hre_real *(ft(1)+thr)*sinp_hr[ww]);
      std::complex<ft> const itr_lr(lre_real * (ft(1)-tlr)*cosp_lr[ww], lre_real *(ft(1)+tlr)*sinp_lr[ww]);


	  
      // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
      
      std::complex<ft> idlr_dcl = dpsi_lr[ww]*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr[ww],(ft(1)+tlr)*cosp_lr[ww]) -
					       (ft(2)*flr*sinp_lr[ww]*cosp_lr[ww]/denom_lr) * itr_lr)* zern4_Wnm;
    
    
    
      // --- derivative with respect to CL --- //
	  
      dlr_dcl[ww] += idlr_dcl; // integration weight is already included
      
      
      
      // --- derivative with respect to CH --- //
      
      dhr_dch[ww] += zern4_Wnm * ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr[ww],(ft(1)+thr)*cosp_hr[ww]) -
				   (2*fhr*sinp_hr[ww]*cosp_hr[ww]/denom_hr) * itr_hr)*dpsi_hr[ww]);
      
      dlr_dch[ww] += idlr_dcl * decl_ech; // integration weight is already included
      
      
      
      // --- derivative with respect to HR --- //
      
      dhr_dthr[ww] += zern4_Wnm * (itr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr[ww])/denom_hr)) +
				   hre_real * (std::complex<ft>(-cosp_hr[ww],sinp_hr[ww])));
      
      
      
      // --- derivative with respect to LR --- //
      
      dlr_dtlr[ww] += zern4_Wnm * (itr_lr * (cLRE - (dflr_derl * mth::SQ(sinp_lr[ww])/denom_lr)) +
				   lre_real * (std::complex<ft>(-cosp_lr[ww],sinp_lr[ww])));
      
      
      tr_lr[ww] += zern4_Wnm * itr_lr;
      tr_hr[ww] += zern4_Wnm * itr_hr;
      
    } // ww
    
  } // n

  
  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;
  ft* const dltr_dech = dtr + 4*N1;
  
  
  // --- now propagate the intensity transmission profile --- //

  for(int ww=0; ww<N1; ++ww){  	  
    htr[ww] = (tr_hr[ww]*std::conj(tr_hr[ww])).real();
    ltr[ww] = (tr_lr[ww]*std::conj(tr_lr[ww])).real();

    
    // --- apply the chain rule to the derivative of tr --- //
    
    dtr_derl[ww] = (tr_lr[ww]*std::conj(dlr_dtlr[ww]) + std::conj(tr_lr[ww])*dlr_dtlr[ww]).real();
    dtr_derh[ww] = (tr_hr[ww]*std::conj(dhr_dthr[ww]) + std::conj(tr_hr[ww])*dhr_dthr[ww]).real();
    dtr_decl[ww] = (tr_lr[ww]*std::conj(dlr_dcl[ww])  + std::conj(tr_lr[ww])*dlr_dcl[ww]).real();
    dtr_dech[ww] = (tr_hr[ww]*std::conj(dhr_dch[ww])  + std::conj(tr_hr[ww])*dhr_dch[ww]).real();
    dltr_dech[ww] = (tr_lr[ww]*std::conj(dlr_dch[ww])  + std::conj(tr_lr[ww])*dlr_dch[ww]).real();
  }

  // --- clean-up --- //
  
  delete [] tr_hr;
  delete [] sinp_lr;
  
  
  // --- Area normalization LRE --- //
  
  if(normalize_ltr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += ltr[ii];
      sum1+= dtr_derl[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_decl[ii]  *= sum;
      dltr_dech[ii] *= sum;
      dtr_derl[ii] = dtr_derl[ii]*sum - sum2*ltr[ii];
      ltr[ii] *= sum; 
    }
  }
  
  // --- Area normalization HRE --- //

  if(normalize_htr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += htr[ii];
      sum1+= dtr_derh[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_dech[ii] *= sum;
      dtr_derh[ii]  = dtr_derh[ii]*sum - sum2*htr[ii];
      htr[ii]      *= sum; 
    }
  }

}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_complex(int const N1, const ft* const tw,
				    ft* const tr,
				    ft const erh, ft const erl,
				    ft const ech, ft const ecl,
				    ft const angle, bool const normalize)const
{

  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);


  // --- precompute quantities --- //
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  
  // --- wavelength loop --- //
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = ft(1) / ( tw[ii] + cw);
    ft const psi_lr = plr * wav1;
    ft const psi_hr = phr * wav1;
    
    ft const sinp_hr = std::sin(psi_hr);
    ft const sinp_lr = std::sin(psi_lr);
    ft const cosp_hr = std::cos(psi_hr);
    ft const cosp_lr = std::cos(psi_lr);

    ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr));
    ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr));
    
    std::complex<ft> const tr_nu = std::complex<ft>(hre_real * (ft(1)-thr)*cosp_hr, hre_real *(ft(1)+thr)*sinp_hr) *
      std::complex<ft>(lre_real * (ft(1)-tlr)*cosp_lr, lre_real *(ft(1)+tlr)*sinp_lr);

    tr[ii] = (tr_nu * std::conj(tr_nu)).real();
  }


  if(normalize){

    ft suma = ft(0);
    
    for(int ii=0; ii<N1; ++ii)
      suma += tr[ii];

    suma = ft(1) / suma;
    
    
    // --- area normalize --- //
    
    for(int ii=0; ii<N1; ++ii)
      tr[ii] *= suma;
  }
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_complex_der(int const N1, const ft* const tw,
					ft* const tr, ft* const dtr,
					ft const erh, ft const erl,
					ft const ech, ft const ecl,
					ft const angle, bool const normalize)const
{

  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);


  // --- precompute quantities --- //
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error

  ft const dfhr_derh = fhr * (ft(1) / thr + ft(2) / (ft(1)-thr)) * dthr_derh;
  ft const dflr_derl = flr * (ft(1) / tlr + ft(2) / (ft(1)-tlr)) * dtlr_derl;
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  ft const dphr_dech = ca;
  ft const dplr_decl = ca;
  
  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);


  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;

  
  // --- wavelength loop --- //
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = ft(1) / ( tw[ii] + cw);
    ft const psi_lr = plr * wav1;
    ft const psi_hr = phr * wav1;
    ft const dpsi_lr =  dplr_decl * wav1;
    ft const dpsi_hr =  dphr_dech * wav1;
    
    ft sinp_hr, sinp_lr, cosp_hr, cosp_lr;
    sincos(psi_lr, &sinp_lr, &cosp_lr), sincos(psi_hr, &sinp_hr, &cosp_hr);

    ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr);     
    ft const lre_real = cLRE / denom_lr;
    
    
    ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr);
    ft const hre_real = cHRE / denom_hr;
    
    
    std::complex<ft> const tr_hr(hre_real * (ft(1)-thr)*cosp_hr, hre_real *(ft(1)+thr)*sinp_hr);
    std::complex<ft> const tr_lr(lre_real * (ft(1)-tlr)*cosp_lr, lre_real *(ft(1)+tlr)*sinp_lr);
    std::complex<ft> const tr_nu = tr_lr * tr_hr;
    
	  
    // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
      
    std::complex<ft> const dlr_dcl = tr_hr * dpsi_lr*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr,(ft(1)+tlr)*cosp_lr) -
						      (ft(2)*flr*sinp_lr*cosp_lr/denom_lr) * tr_lr);

    
    
    // --- derivative with respect to CH --- //
    
    std::complex<ft>  dhr_dch =  ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr,(ft(1)+thr)*cosp_hr) -
				   (2*fhr*sinp_hr*cosp_hr/denom_hr) * tr_hr)*dpsi_hr);
      

    std::complex<ft> const dtr_dch = dhr_dch*tr_lr + dlr_dcl * decl_ech ; // *tr_hr already included in dlr_dcl
      
      
    // --- derivative with respect to HR --- //
      
    std::complex<ft> const dhr_dthr = tr_lr * (tr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr)/denom_hr)) +
				       hre_real * (std::complex<ft>(-cosp_hr,sinp_hr)));
      
      
      
      // --- derivative with respect to LR --- //
      
    std::complex<ft> const dlr_dtlr = tr_hr * (tr_lr * (cLRE - (dflr_derl * mth::SQ(sinp_lr)/denom_lr)) +
				       lre_real * (std::complex<ft>(-cosp_lr,sinp_lr)));
      
      
    
    
    tr[ii] = (tr_nu * std::conj(tr_nu)).real();
    dtr_derl[ii] = (tr_nu*std::conj(dlr_dtlr) + std::conj(tr_nu)*dlr_dtlr).real(); 
    dtr_derh[ii] = (tr_nu*std::conj(dhr_dthr) + std::conj(tr_nu)*dhr_dthr).real();
    dtr_decl[ii] = (tr_nu*std::conj(dlr_dcl)  + std::conj(tr_nu)*dlr_dcl).real();
    dtr_dech[ii] = (tr_nu*std::conj(dtr_dch)  + std::conj(tr_nu)*dtr_dch).real();
    
  }


  // --- Area normalization of the profile and derivatives? --- //

  if(normalize){
    ft sum = ft(0);
    ft sum1 = ft(0);
    ft sum2 = ft(0);
    
    
    for(int ii=0; ii<N1; ++ii){
      sum += tr[ii];
      sum1+= dtr_derh[ii];
      sum2+= dtr_derl[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum3 = sum*sum;
    
    for(int ii=0; ii<N1; ++ii){
      
      dtr_dech[ii] *= sum;
      dtr_decl[ii] *= sum;
      dtr_derh[ii] = dtr_derh[ii]*sum - sum3*tr[ii]*sum1;	
      dtr_derl[ii] = dtr_derl[ii]*sum - sum3*tr[ii]*sum2;
      tr[ii] *= sum;
    }
  }
  
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_complex_individual(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
					       ft const erh, ft const erl, ft const ech,
					       ft const ecl,  ft const angle, bool const normalize_ltr,
					       bool const normalize_htr)const
{
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);


  // --- precompute quantities --- //
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);

  
  // --- wavelength loop --- //
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = ft(1) / ( tw[ii] + cw);
    ft const psi_lr = plr * wav1;
    ft const psi_hr = phr * wav1;
    
    ft sinp_hr, sinp_lr, cosp_hr, cosp_lr;
    sincos(psi_lr, &sinp_lr, &cosp_lr), sincos(psi_hr, &sinp_hr, &cosp_hr);

    ft const lre_real = cLRE / (ft(1) + flr * mth::SQ(sinp_lr));
    ft const hre_real = cHRE / (ft(1) + fhr * mth::SQ(sinp_hr));
    
    std::complex<ft> const tr_hr(hre_real * (ft(1)-thr)*cosp_hr, hre_real *(ft(1)+thr)*sinp_hr);
    std::complex<ft> const tr_lr(lre_real * (ft(1)-tlr)*cosp_lr, lre_real *(ft(1)+tlr)*sinp_lr);

    htr[ii] = (tr_hr * std::conj(tr_hr)).real();
    ltr[ii] = (tr_lr * std::conj(tr_lr)).real();
    
  }

  
  // --- Profiles normalization --- //
  
  if(normalize_ltr){
    ft sum = 0;
    for(int ii=0; ii<N1; ++ii){
      sum += ltr[ii];
    }
    
    sum = ft(1) / sum;
    
    for(int ii=0; ii<N1; ++ii){
      ltr[ii] *= sum;
    }
  }

  if(normalize_htr){
    ft sum = 0;
    for(int ii=0; ii<N1; ++ii){
      sum += htr[ii];
    }
    
    sum = ft(1) / sum;
    
    for(int ii=0; ii<N1; ++ii){
      htr[ii] *= sum;
    }
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_complex_individual_der(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
						   ft* const dtr, ft const erh, ft const erl, ft const ech,
						   ft const ecl,  ft const angle, bool const normalize_ltr,
						   bool const normalize_htr)const
{
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  
  
  // --- precompute quantities --- //
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const dfhr_derh = fhr * (ft(1) / thr + ft(2) / (ft(1)-thr)) * dthr_derh;
  ft const dflr_derl = flr * (ft(1) / tlr + ft(2) / (ft(1)-tlr)) * dtlr_derl;
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  ft const dphr_dech = ca;
  ft const dplr_decl = ca;
  
  ft const cLRE = ft(1)/(ft(1)-tlr);
  ft const cHRE = ft(1)/(ft(1)-thr);


  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;
  ft* const dltr_dech = dtr + 4*N1;

  
  // --- wavelength loop --- //
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = ft(1) / ( tw[ii] + cw);
    ft const psi_lr = plr * wav1;
    ft const psi_hr = phr * wav1;
    ft const dpsi_lr =  dplr_decl * wav1;
    ft const dpsi_hr =  dphr_dech * wav1;
    
    ft sinp_hr, sinp_lr, cosp_hr, cosp_lr;
    sincos(psi_lr, &sinp_lr, &cosp_lr), sincos(psi_hr, &sinp_hr, &cosp_hr);

    ft const denom_lr =  ft(1) + flr * mth::SQ(sinp_lr);     
    ft const lre_real = cLRE / denom_lr;
    
    
    ft const denom_hr =  ft(1) + fhr * mth::SQ(sinp_hr);
    ft const hre_real = cHRE / denom_hr;
    
    
    std::complex<ft> const tr_hr(hre_real * (ft(1)-thr)*cosp_hr, hre_real *(ft(1)+thr)*sinp_hr);
    std::complex<ft> const tr_lr(lre_real * (ft(1)-tlr)*cosp_lr, lre_real *(ft(1)+tlr)*sinp_lr);
    
	  
    // --- Store dlr_dcl to propagate the dcl_dch, given that ech sets the zero point --- //
      
    std::complex<ft> const dlr_dcl = dpsi_lr*(lre_real*std::complex<ft>((tlr-ft(1))*sinp_lr,(ft(1)+tlr)*cosp_lr) -
						  (ft(2)*flr*sinp_lr*cosp_lr/denom_lr) * tr_lr);
    
    std::complex<ft> const dlr_dch =  dlr_dcl * decl_ech;
    
    
    // --- derivative with respect to CH --- //
    
    std::complex<ft>  dhr_dch =  ((hre_real*std::complex<ft>((thr-ft(1))*sinp_hr,(ft(1)+thr)*cosp_hr) -
					(2*fhr*sinp_hr*cosp_hr/denom_hr) * tr_hr)*dpsi_hr);
      

    std::complex<ft> const dtr_dch = dhr_dch*tr_lr + dlr_dcl * decl_ech * tr_hr;
      
      
      // --- derivative with respect to HR --- //
      
    std::complex<ft> const dhr_dthr = (tr_hr * (cHRE - (dfhr_derh * mth::SQ(sinp_hr)/denom_hr)) +
				       hre_real * (std::complex<ft>(-cosp_hr,sinp_hr)));
      
      
      
      // --- derivative with respect to LR --- //
      
    std::complex<ft> const dlr_dtlr = (tr_lr * (cLRE - (dflr_derl * mth::SQ(sinp_lr)/denom_lr)) +
				       lre_real * (std::complex<ft>(-cosp_lr,sinp_lr)));
      
      
    
    htr[ii] = (tr_hr*std::conj(tr_hr)).real();
    ltr[ii] = (tr_lr*std::conj(tr_lr)).real();

    
    // --- apply the chain rule to the derivative of tr --- //
    
    dtr_derl[ii] = (tr_lr*std::conj(dlr_dtlr) + std::conj(tr_lr)*dlr_dtlr).real();
    dtr_derh[ii] = (tr_hr*std::conj(dhr_dthr) + std::conj(tr_hr)*dhr_dthr).real();
    dtr_decl[ii] = (tr_lr*std::conj(dlr_dcl)  + std::conj(tr_lr)*dlr_dcl).real();
    dtr_dech[ii] = (tr_hr*std::conj(dhr_dch)  + std::conj(tr_hr)*dhr_dch).real();
    dltr_dech[ii] = (tr_lr*std::conj(dlr_dch)  + std::conj(tr_lr)*dlr_dch).real();
    
  }

  
  // --- Area normalization LRE --- //
  
  if(normalize_ltr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += ltr[ii];
      sum1+= dtr_derl[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_decl[ii]  *= sum;
      dltr_dech[ii] *= sum;
      dtr_derl[ii] = dtr_derl[ii]*sum - sum2*ltr[ii];
      ltr[ii] *= sum; 
    }
  }
  
  // --- Area normalization HRE --- //

  if(normalize_htr){
    ft sum = ft(0);
    ft sum1 = ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += htr[ii];
      sum1+= dtr_derh[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_dech[ii] *= sum;
      dtr_derh[ii]  = dtr_derh[ii]*sum - sum2*htr[ii];
      htr[ii]      *= sum; 
    }
  }

}

// ********************************************************************* //
