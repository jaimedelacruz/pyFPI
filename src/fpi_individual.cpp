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

#include "fpi.hpp"
#include "fpi_helper.hpp"

// ********************************************************************* //


void fpi::FPI::dual_fpi_conv_individual_der(int const N1, const ft* const tw,
					    ft* const htr, ft* const ltr, ft* const dtr,
					    ft const erh, ft const erl,
					    ft const ech, ft const ecl,
					    bool const normalize_ltr,
					    bool const normalize_htr)const
{

  
  // --- zero result arrays --- //
  
  std::memset(htr,0,N1*sizeof(ft));
  std::memset(ltr,0,N1*sizeof(ft));
  std::memset(dtr,0,5*N1*sizeof(ft));
  
  

  // --- assign pointers to each derivative --- //
  
  ft* const __restrict__ dtr_derh = dtr;
  ft* const __restrict__ dtr_derl = dtr+1*N1;
  ft* const __restrict__ dtr_dech = dtr+2*N1;
  ft* const __restrict__ dtr_decl = dtr+3*N1;
  ft* const __restrict__ dltr_dech = dtr+4*N1;

  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);
  
  
  
  // --- Finesse --- //

  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  
  ft const dfhr_derh = fhr * (ft(1) / thr + ft(2) / (ft(1)-thr)) * dthr_derh;
  ft const dflr_derl = flr * (ft(1) / tlr + ft(2) / (ft(1)-tlr)) * dtlr_derl;
    
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  

  
  // --- Loop over angle and wavelength positions --- //
  
  for(int nn=0; nn<fpi::NRAYS; ++nn){
    
    ft const plr = calp[nn]*(lc+ecl_ech);
    ft const phr = calp[nn]*(hc+ech);
    
    ft const dplr_decl = calp[nn];
    ft const dphr_dech = calp[nn];

    
    for(int ww=0; ww < N1; ++ww){

      // --- apply the chain rule to the profile variables --- //
      
      ft const wav1 = tw[ww] + cw + BlueShift;

      ft const phr_wav1_ech = phr / wav1;
      ft const plr_wav1_ecl = plr / wav1;
      
      ft const lre_sin  = std::sin(plr_wav1_ecl);
      ft const hre_sin  = std::sin(phr_wav1_ech);

      ft const lre_2sincos = ft(2) * std::cos(plr_wav1_ecl) * lre_sin;
      ft const hre_2sincos = ft(2) * std::cos(phr_wav1_ech) * hre_sin;

      ft const tr_lre = ft(1) / ( ft(1) + flr * mth::SQ(lre_sin) );  
      ft const tr_hre = ft(1) / ( ft(1) + fhr * mth::SQ(hre_sin) );
      
      ft const dtr_lre_derl = - mth::SQ(tr_lre) * dflr_derl * mth::SQ(lre_sin);
      ft const dtr_hre_derh = - mth::SQ(tr_hre) * dfhr_derh * mth::SQ(hre_sin);

      ft const dtr_hre_dech =  -mth::SQ(tr_hre) * fhr * hre_2sincos * (dphr_dech / wav1);
      ft const dtr_lre_decl =  -mth::SQ(tr_lre) * flr * lre_2sincos * (dplr_decl / wav1);      

      
      // --- populate output variables with the profile and its four derivatives --- //
      
      htr[ww] += tr_hre * wng[nn];
      ltr[ww] += tr_lre * wng[nn];
      
      dtr_derh[ww] += dtr_hre_derh * wng[nn];
      dtr_derl[ww] += dtr_lre_derl * wng[nn];

      dtr_dech[ww] += dtr_hre_dech * wng[nn];
      dtr_decl[ww] += dtr_lre_decl * wng[nn];
      dltr_dech[ww]+= dtr_lre_decl * decl_ech * wng[nn];

    } // ww
  } // nn


  // --- Normalize LRE? --- //
  
  if(normalize_ltr){
    ft sum = ft(0);
    ft sum1= ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += ltr[ii];
      sum1+= dtr_derl[ii];
    }

    sum = ft(1) / sum;
    ft const sum2 = sum*sum * sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_derl[ii]   = sum*dtr_derl[ii] - sum2*ltr[ii];
      dtr_decl[ii]  *= sum;
      dltr_dech[ii] *= sum;
      ltr[ii]       *= sum;
    }
  }

  
  // --- Normalize HRE? --- //

  if(normalize_htr){
    ft sum = ft(0);
    ft sum1= ft(0);
    
    for(int ii=0; ii<N1; ++ii){
      sum += htr[ii];
      sum1+= dtr_derh[ii];
    }
    
    sum = ft(1) / sum;
    ft const sum2 = sum*sum*sum1;
    
    for(int ii=0; ii<N1; ++ii){
      dtr_derh[ii] =  sum*dtr_derh[ii] - sum2*htr[ii];
      dtr_dech[ii] *= sum;
      htr[ii] *= sum;

    }
  }
  
}


// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_individual(int const N1, const ft* const tw,
					ft* const htr, ft* const ltr,
					ft const erh, ft const erl,
					ft const ech, ft const ecl,
					bool const normalize_ltr,
					bool const normalize_htr)const
{

  // --- zero result array --- //
  
  std::memset(htr,0,N1*sizeof(ft));
  std::memset(ltr,0,N1*sizeof(ft));
  
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  

  
  // --- Loop over angle and wavelength positions --- //
  
  for(int nn=0; nn<fpi::NRAYS; ++nn){
    
    ft const plr = calp[nn]*(lc+ecl_ech);
    ft const phr = calp[nn]*(hc+ech);
    ft const iwng = wng[nn];
    
    for(int ww=0; ww < N1; ++ww){
      ft const wav1 = tw[ww] + cw + BlueShift;
      
      ltr[ww] += iwng / ( ft(1) + flr * mth::SQ(std::sin(plr / wav1)) );
      htr[ww] += iwng / ( ft(1) + fhr * mth::SQ(std::sin(phr / wav1)) );
      
    } // ww
  } // nn


  
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

void fpi::FPI::dual_fpi_full_individual(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
					ft const erh, ft const erl, ft const ech,
					ft const ecl,  bool const normalize_ltr,
					bool const normalize_htr)const
{

  std::memset(htr,0,N1*sizeof(ft));
  std::memset(ltr,0,N1*sizeof(ft));

  
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


  // --- construct the profile with the angle integral --- //
  
  for(int n=0; n<NRAYS_LR; ++n){
    for(int m=0; m<NRAYS_HR; ++m){
      
      ft const ibetah = n_betah(n,m);
      if(ibetah < ft(1.e-14))
	continue;
      
      for(int ww=0; ww<N1; ++ww){
	
	htr[ww] += ibetah / (ft(1) + fhr * sin2p_hr(m,ww));
	ltr[ww] += ibetah / (ft(1) + flr * sin2p_lr(n,ww));
      }
    }
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

void fpi::FPI::dual_fpi_full_individual_der(int const N1, const ft* const tw, ft* const htr, ft* const ltr,
					    ft* const dtr, ft const erh, ft const erl, ft const ech,
					    ft const ecl,  bool const normalize_ltr,
					    bool const normalize_htr)const
{

  std::memset(htr,0,N1*sizeof(ft));
  std::memset(ltr,0,N1*sizeof(ft));
  std::memset(dtr,0,5*N1*sizeof(ft));

  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);

  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);
  ft const dfhr_derh = fhr * (ft(1) / thr + ft(2) / (ft(1)-thr)) * dthr_derh;
  ft const dflr_derl = flr * (ft(1) / tlr + ft(2) / (ft(1)-tlr)) * dtlr_derl;
  
  
  
  // --- get sin2p and its derivatives relative to the input cavity separation --- //

  fpi::Arr2D<ft> dsin2p_hr, dsin2p_lr;
  fpi::Arr2D<ft> sin2p_hr = fpi::get_psi2_der(N1,cw+BlueShift,tw,hc+ech,betah_hr,dsin2p_hr);

  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  fpi::Arr2D<ft> sin2p_lr = fpi::get_psi2_der(N1,cw+BlueShift,tw,lc_tilted+ecl_ech,betah_lr,dsin2p_lr);

  

  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;
  ft* const dltr_dech = dtr + 4*N1;


  
  // --- construct the profile with the angle integral --- //
  
  for(int n=0; n<NRAYS_LR; ++n){
    for(int m=0; m<NRAYS_HR; ++m){
      
      ft const ibeta = n_betah(n,m);
      
      if(ibeta < ft(1.e-14))
	continue; // skip this ray, it has weight zero!

      
      for(int ww=0; ww<N1; ++ww){

	ft const tr_hr = ft(1) / (ft(1) + fhr * sin2p_hr(m,ww));
	ft const tr_lr = ft(1) / (ft(1) + flr * sin2p_lr(n,ww));
	
	ft const dtr_lr_derl = - mth::SQ(tr_lr) * dflr_derl * sin2p_lr(n,ww);
	ft const dtr_hr_derh = - mth::SQ(tr_hr) * dfhr_derh * sin2p_hr(m,ww);
	
	ft const dtr_lr_decl = mth::SQ(tr_lr) * flr * dsin2p_lr(n,ww);
	ft const dtr_lr_dech = dtr_lr_decl * decl_ech;

	ft const dtr_hr_dech = mth::SQ(tr_hr) * fhr * dsin2p_hr(m,ww);


	dtr_derh[ww] += dtr_hr_derh * ibeta;
	dtr_derl[ww] += dtr_lr_derl * ibeta;

	dtr_decl[ww] += dtr_lr_decl*ibeta;
	dtr_dech[ww] += dtr_hr_dech*ibeta;
	dltr_dech[ww]+= dtr_lr_dech*ibeta;
	  
	ltr[ww] += tr_lr * ibeta;
	htr[ww] += tr_hr * ibeta;
      }
    }
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

void fpi::FPI::dual_fpi_ray_individual(int const N1, const ft* const tw,
				       ft* const htr, ft* const ltr,
				       ft const erh, ft const erl,
				       ft const ech, ft const ecl,
				       ft const angle, bool const normalize_ltr,
				       bool const normalize_htr)const
{
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);

  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  
  
  // --- transmission profiles --- //

  ft hsuma = ft(0);
  ft lsuma = ft(0);
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = tw[ii] + cw;
    
    ltr[ii] = ft(1) / ( ft(1) + flr * mth::SQ(std::sin(plr / wav1)) );
    htr[ii] = ft(1) / ( ft(1) + fhr * mth::SQ(std::sin(phr / wav1)) );

    hsuma += htr[ii];
    lsuma += ltr[ii];
  }

  
  // --- area normalize HRE--- //

  if(normalize_htr){
    hsuma = ft(1) / hsuma;
    for(int ii=0; ii<N1; ++ii){
      htr[ii] *= hsuma;
    }
  }


  // --- area normalize LRE--- //

  if(normalize_ltr){
    lsuma = ft(1) / lsuma;
    for(int ii=0; ii<N1; ++ii){
      ltr[ii] *= lsuma;
    }
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_individual_der(int const N1, const ft* const tw,
					   ft* const htr, ft* const ltr, ft* const dtr,
					   ft const erh, ft const erl,
					   ft const ech, ft const ecl,
					   ft const angle, bool const normalize_ltr,
					   bool const normalize_htr)const
{

  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;
  ft* const dltr_dech = dtr + 4*N1;

  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;
  
  constexpr ft const dthr_derh = ft(1);
  constexpr ft const dtlr_derl = ft(1);
  
  
  // --- Finesse --- //
  
  ft const fhr = ft(4) * thr / mth::SQ(ft(1) - thr);
  ft const flr = ft(4) * tlr / mth::SQ(ft(1) - tlr);

  ft const dfhr_derh = fhr * (ft(1) / thr + ft(2) / (ft(1)-thr)) * dthr_derh;
  ft const dflr_derl = flr * (ft(1) / tlr + ft(2) / (ft(1)-tlr)) * dtlr_derl;
  
  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const ca = two_pi * std::cos(angle);
  
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  ft const dphr_dech = ca;
  ft const dplr_decl = ca;
  

  
  // --- transmission --- //

  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = tw[ii] + cw;

    ft const phr_wav1_ech = phr / wav1;
    ft const plr_wav1_ecl = plr / wav1;

    ft const lre_sin  = std::sin(plr_wav1_ecl);
    ft const hre_sin  = std::sin(phr_wav1_ech);
    
    ft const hre_2sincos = ft(2) * std::cos(phr_wav1_ech) * hre_sin;
    ft const lre_2sincos = ft(2) * std::cos(plr_wav1_ecl) * lre_sin;

    ft const tr_lre = ft(1) / (ft(1) + flr * mth::SQ(lre_sin));
    ft const tr_hre = ft(1) / (ft(1) + fhr * mth::SQ(hre_sin));
    
    ft const dtr_lre_derl = - mth::SQ(tr_lre) * dflr_derl * mth::SQ(lre_sin);
    ft const dtr_hre_derh = - mth::SQ(tr_hre) * dfhr_derh * mth::SQ(hre_sin);
    
    ft const dtr_hre_dech = -mth::SQ(tr_hre) * fhr * hre_2sincos * (dphr_dech / wav1);
    ft const dtr_lre_decl = -mth::SQ(tr_lre) * flr * lre_2sincos * (dplr_decl / wav1);
    
    dltr_dech[ii] = dtr_lre_decl * decl_ech;
    
    htr[ii] = tr_hre;
    ltr[ii] = tr_lre;
    
    dtr_derh[ii] = dtr_hre_derh;
    dtr_derl[ii] = dtr_lre_derl;
    
    dtr_dech[ii] = dtr_hre_dech;
    dtr_decl[ii] = dtr_lre_decl;
     
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
      
      dltr_dech[ii] *= sum;
      dtr_decl[ii] *= sum;
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
      dtr_derh[ii] = dtr_derh[ii]*sum - sum2*htr[ii];
      htr[ii] *= sum;
    } 
  }
  
}

// ********************************************************************* //
