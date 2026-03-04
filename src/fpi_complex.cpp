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
  constexpr const int Ndef = 51;
  constexpr const ft max_defocus_mm = 6.0;

  
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
    tw[ii] = (ii-N/2)*0.004;


  // --- for each defocus length, tests max peak transmission --- //

  int itmax = 0;
      
  for(int id = 0; id<Ndef; ++id){

    // --- fill in zern4 terms --- //

    for(int ii=0; ii<NRAYS_HR; ++ii)
      this->zern4[ii] = std::exp(std::complex<ft>(0.0, -defocus[id]*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0)));

    
    // --- calculate profile --- //

    dual_fpi_full_complex(N,tw.data(), tr.data(), 0.0,0.0,0.0,0.0,false);

    ft ipmax = tr[0], tmax = tr[0];
    
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
    std::cerr<<std::format("angle={:f}\n", angles[ii]);
    ft const zernike_defocus = defocus_final*std::sqrt(3.0)*(2*mth::SQ(angles[ii])-1.0);
    this->zern4[ii] = std::exp(std::complex<ft>(0.0, -zernike_defocus));
  }

  std::cerr<<std::format("fpi::FPI::optimize_Zernike: cZer={:.4f} mm, tmax={:.3f}\n", defocus_final/defocus_scl, pmax[itmax+1]);
  
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

	ft const w_mn = this->n_betah(n,m);
	
	for(int ww=0; ww<N1; ++ww){  	  

	  // --- now multiply the two transmission profiles --- //
	  
	  tr_nu[ww] += (tr_hr[ww] * tr_lr[ww] * this->zern4[m])*w_mn;
	  
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
