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

#include "fpi.hpp"
#include "fpi_helper.hpp"
#include "math.hpp"

// ********************************************************************* //

void fpi::FPI::dual_fpi_full_complex()
{
  
  std::memset(tr,0,N1*sizeof(ft));

  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  
  // --- get sin2p --- //

  fpi::Arr2D<ft> sin2p_hr = fpi::get_psi2(N1,cw+BlueShift,tw,hc+ech,betah_hr);


  ft const ecl_ech = ecl + ech*(lc/hc); // include the HR cavity error
  fpi::Arr2D<ft> sin2p_lr = fpi::get_psi2(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr);
 
  ft const T2_hr = mth::SQ(ft(1) - thr);
  ft const T2_lr = mth::SQ(ft(1) - lhr);
  
  
  // --- construct the profile with the angle integral --- //

  for(int ww=0; ww<N1; ++ww){

    std::complex<ft> tr_nu(0.0,0.0);
    
    
    for(int n=0; n<NRAYS_LR; ++n){
      for(int m=0; m<NRAYS_HR; ++m){
	
	ft const ibetah = n_betah(n,m);

	
	std::complex<ft> tr_hr = T2_hr * (ft(1) - mth::SQ(thr) );
	
	//ft const tr_hr = ft(1) / (ft(1) + fhr * sin2p_hr(m,ww));
	//ft const tr_lr = ft(1) / (ft(1) + flr * sin2p_lr(n,ww));
	
	  
	  tr[ww] += tr_hr * tr_lr * ibetah;
      }
    }
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
