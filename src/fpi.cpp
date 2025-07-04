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

fpi::FPI::~FPI(){};

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv(int const N1, const ft* const tw, ft* const tr,
			     ft const erh, ft const erl, ft const ech,
			     ft const ecl, bool const normalize)const
{
  // --- zero result array --- //
  
  std::memset(tr,0,N1*sizeof(ft));
  
  
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
    
    for(int ww=0; ww < N1; ++ww){
      ft const wav1 = tw[ww] + cw + BlueShift;
      
      ft const tr_lre = ft(1) / ( ft(1) + flr * mth::SQ(std::sin(plr / (wav1))) );
      ft const tr_hre = ft(1) / ( ft(1) + fhr * mth::SQ(std::sin(phr / (wav1))) );

      tr[ww] += (tr_lre * tr_hre) * wng[nn];
      
    } // ww
  } // nn

  
  // --- Area normalization? --- //

  if(normalize){
    ft sum = ft(0);
    for(int ii=0; ii<N1; ++ii)
      sum += tr[ii];
    
    sum = ft(1) / sum;
    for(int ii=0; ii<N1; ++ii)
      tr[ii] *= sum;
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_conv_der(int const N1, const ft* const tw,
				 ft* const tr, ft* const dtr,
				 ft const erh, ft const erl,
				 ft const ech, ft const ecl, bool const normalize)const
{


  // --- assign pointers to each derivative --- //
  
  ft* const __restrict__ dtr_derh = dtr;
  ft* const __restrict__ dtr_derl = dtr+1*N1;
  ft* const __restrict__ dtr_dech = dtr+2*N1;
  ft* const __restrict__ dtr_decl = dtr+3*N1;

  
    // --- zero result arrays --- //
  
  std::memset(tr,0,N1*sizeof(ft));
  std::memset(dtr,0,4*N1*sizeof(ft));
  
  
  // --- Total reflectivity --- //
  
  ft const thr = hr + erh;
  ft const tlr = lr + erl;

  
  constexpr ft const dthr_derh = 1.0;
  constexpr ft const dtlr_derl = 1.0;
  
  
  
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
      ft const lre_2sincos = ft(2) * std::cos(plr_wav1_ecl) * lre_sin;

      ft const hre_sin  = std::sin(phr_wav1_ech);
      ft const hre_2sincos = ft(2) * std::cos(phr_wav1_ech) * hre_sin;

      ft const tr_lre = ft(1) / ( ft(1) + flr * mth::SQ(lre_sin) );  
      ft const tr_hre = ft(1) / ( ft(1) + fhr * mth::SQ(hre_sin) );
      
      ft const dtr_lre_derl = - mth::SQ(tr_lre) * dflr_derl * mth::SQ(lre_sin);
      ft const dtr_hre_derh = - mth::SQ(tr_hre) * dfhr_derh * mth::SQ(hre_sin);

      ft const dtr_hre_dech = -mth::SQ(tr_hre) * fhr * hre_2sincos * (dphr_dech / wav1);
      ft const dtr_lre_decl = -mth::SQ(tr_lre) * flr * lre_2sincos * (dplr_decl / wav1);

      ft const dtr_lr_dech = dtr_lre_decl * decl_ech;

      
      // --- populate output variables with the profile and its four derivatives --- //
      
      tr[ww] += (tr_lre * tr_hre) * wng[nn];
      
      dtr_derh[ww] += (dtr_hre_derh * tr_lre + tr_hre*dtr_lr_dech) * wng[nn];
      dtr_derl[ww] += dtr_lre_derl * tr_hre * wng[nn];

      dtr_dech[ww] += dtr_hre_dech * tr_lre * wng[nn];
      dtr_decl[ww] += dtr_lre_decl * tr_hre * wng[nn];      
    } // ww
  } // nn


  // --- Normalization, apply the chain rule to the derivative of the reflectivity --- //


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

void fpi::FPI::set_reflectivities(ft const ihr, ft const ilr)
{
  // --- Only replace reflectivities if they are set to positive values --- //
  
  if(ilr > 0.0) lr = ilr;
  if(ihr > 0.0) hr = ihr;
}


// ********************************************************************* //

fpi::FPI::FPI(ft const icw, ft const iFR, ft const shr, ft const slr, int const iNRAYS_HR, int const iNRAYS_LR):
  cw(icw), FR(iFR), hc(0), lc(0), lc_tilted(0), hfsr(0), lfsr(0), lr(0), hr(0), BlueShift(0),\
  NRAYS_HR(iNRAYS_HR), NRAYS_LR(iNRAYS_LR), calp{}, wng{}, 
  n_betah(), betah_lr(iNRAYS_LR), betah_hr(iNRAYS_HR),
  convolver(nullptr), convolver2(nullptr)
{
  constexpr ft const n_hr = 1.0; // air
  constexpr ft const n_lr = 1.0; // air


  // --- Init the pupil histogram --- //

  constexpr ft const TILT_HR = ft(0);
  constexpr ft const TILT_LR = ft(1); // in units of 0.5/FR;
  
  
  // --- interpolate reflectivity tables at cw --- //
  
  lr = crisp::CRISP_REF_LRE<ft>.interpolate(cw);
  hr = crisp::CRISP_REF_HRE<ft>.interpolate(cw);

  

  // --- center the transmission peaks at cw --- //

  {
    int const nhr = int((0.5+shr*n_hr) / (cw*0.5));
    hc = nhr * cw * 0.5;
    
    int const nlr = int((0.5+slr*n_lr) / (cw*0.5));
    lc = nlr * cw * 0.5;
    lc_tilted = lc / std::cos(TILT_LR/(ft(2)*FR));
  }

  

  // --- adjust angles for conv approximation --- //

  ft swang = ft(0);
  for(int ii=0; ii<fpi::NRAYS; ++ii){
    ft const iang =  std::sqrt(ft(ii) / ft(fpi::NRAYS - 1) * mth::SQ<ft>(ft(0.5)/FR));
    calp[ii] = std::cos(iang) * two_pi;
    
    wng[ii] = (((ii ==0)||(ii==(fpi::NRAYS-1))) ? 0.5 : 1.0);
    swang += wng[ii];
  }
  
  swang = ft(1) / swang;
  for(int ii=0; ii<fpi::NRAYS; ++ii){
    wng[ii] *= swang;
  }
  
  
  // --- save FSR --- //
  
  hfsr = mth::SQ(cw) / (2.0 * hc + cw);
  lfsr = mth::SQ(cw) / (2.0 * lc + cw);

  

  // --- Pre-compute histogram of pupil angles and weights --- //
  
  Arr2D<int> ap = fpi::aperture<int,ft>(fpi::NL,fpi::NR);
  Arr2D<ft> beta_fpi1 = fpi::fp_angles(fpi::NL,ap,FR,TILT_HR) / ft(n_hr);
  Arr2D<ft> beta_fpi2 = fpi::fp_angles(fpi::NL,ap,FR,TILT_LR) / ft(n_lr);
  Arr3D<ft> betah; betah.setZero();

  fpi::hist2D(ap,beta_fpi1,beta_fpi2,NRAYS_HR,NRAYS_LR,n_betah,betah);


  // --- Store rays for each etalon --- //

  for(int ii=0; ii<NRAYS_LR;++ii)
    betah_lr[ii] = betah(1,ii,0);

  for(int ii=0; ii<NRAYS_HR;++ii)
    betah_hr[ii] = betah(0,0,ii);


  
  // --- estimate profile blueshift --- //

  BlueShift = getBlueShift();

  
}

// ********************************************************************* //

ft fpi::FPI::getBlueShift()const
{

  // --- create array to generate a profile --- //
  
  constexpr const int NN = 201;
  constexpr const int N2 = NN/2;

  ft* const tw = new ft[NN]();
  ft* const tr = new ft[NN]();

  for(int ii=0; ii<NN; ++ii){
    tw[ii] = ft(ii-N2)*ft(0.001);
  }

  
  // --- generate the FPI transmission profile --- //
  
  dual_fpi_full(NN, tw, tr,0.0, 0.0, 0.0, 0.0,false);



  // --- Find the maximum, should be close to N2, but not exactly --- //
  
  ft center_of_mass = 0.0;
  ft sum = 0.0;

  int imax = 0;
  ft vmax = tr[0];
  
  for(int ii=1; ii<NN; ++ii){
    if(vmax < tr[ii]){
      imax = ii;
      vmax = tr[ii];
    }
  }

  // --- integrate symmetrically to get the center of mass --- //
  
  for(int ii=0; ii<imax; ++ii){
    sum += tr[ii];
    center_of_mass += tr[ii]*tw[ii];

    sum += tr[ii+imax];
    center_of_mass += tr[ii+imax]*tw[ii+imax];
  }  

  
  // ---- Normalize by the area of the profile --- //
  
  center_of_mass /= sum;



  // --- cleanup --- //
  
  delete [] tw;
  delete [] tr;

  
  return center_of_mass;
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray(int const N1, const ft* const tw,
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

  ft const decl_ech = lc/hc;
  ft const ecl_ech = ecl + ech*decl_ech; // include the HR cavity error
  
  ft const ca = two_pi * std::cos(angle);
  ft const phr = (hc+ech) * ca;
  ft const plr = (lc+ecl_ech) * ca;

  
  
  // --- transmission --- //

  ft suma = ft(0);
  
  for(int ii=0; ii<N1; ++ii){
    ft const wav1 = tw[ii] + cw;
    
    ft const tr_lre = ft(1) / ( ft(1) + flr * mth::SQ(std::sin(plr / wav1)) );
    ft const tr_hre = ft(1) / ( ft(1) + fhr * mth::SQ(std::sin(phr / wav1)) );

    tr[ii] = tr_lre*tr_hre;
    suma += tr[ii];
  }

  if(normalize){
    suma = ft(1) / suma;
    
    
    // --- area normalize --- //
    
    for(int ii=0; ii<N1; ++ii)
      tr[ii] *= suma;
  }
  
}

// ********************************************************************* //

void fpi::FPI::dual_fpi_ray_der(int const N1, const ft* const tw,
				ft* const tr, ft* const dtr,
				ft const erh, ft const erl,
				ft const ech, ft const ecl,
				ft const angle, bool const normalize)const
{
  // --- Init pointers for derivatives --- //

  ft* const dtr_derh = dtr;
  ft* const dtr_derl = dtr + 1*N1;
  ft* const dtr_dech = dtr + 2*N1;
  ft* const dtr_decl = dtr + 3*N1;

  
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

    ft const tr_lre = ft(1) / ( ft(1) + flr * mth::SQ(lre_sin));
    ft const tr_hre = ft(1) / ( ft(1) + fhr * mth::SQ(hre_sin));
    
    ft const dtr_lre_derl = - mth::SQ(tr_lre) * dflr_derl * mth::SQ(lre_sin);
    ft const dtr_hre_derh = - mth::SQ(tr_hre) * dfhr_derh * mth::SQ(hre_sin);
    
    ft const dtr_hre_dech = -mth::SQ(tr_hre) * fhr * hre_2sincos * (dphr_dech / wav1);
    ft const dtr_lre_decl = -mth::SQ(tr_lre) * flr * lre_2sincos * (dplr_decl / wav1);

    
    tr[ii] = tr_lre*tr_hre;
    
    dtr_derh[ii] = dtr_hre_derh * tr_lre;
    dtr_derl[ii] = dtr_lre_derl * tr_hre;
    
    dtr_dech[ii] = dtr_hre_dech * tr_lre + tr_hre * dtr_lre_decl * decl_ech;
    dtr_decl[ii] = dtr_lre_decl * tr_hre;
    
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

ft fpi::FPI::getFWHM()const
{
  constexpr const ft n_hre = 1;
  
  ft const Fr = PI * std::sqrt(hr) / (ft(1) - hr);
  return mth::SQ(cw) / (ft(2) * Fr * n_hre * hc);
}

// ********************************************************************* //

ft const fpi::FPI::getFSR()const
{
  return hfsr;
}

// ********************************************************************* //

void fpi::FPI::init_convolver(int const ndata, int const npsf)
{

  convolver = std::make_unique<mth::fftconv1D<ft>>(ndata,npsf);
  
}

// ********************************************************************* //

void fpi::FPI::init_convolver2(int const ndata, int const npsf)
{

  convolver2 = std::make_unique<mth::fftconv1D<ft>>(ndata,npsf);
  
}
// ********************************************************************* //

void fpi::FPI::dual_fpi_full(int const N1, const ft* const tw, ft* const tr,
			     ft const erh, ft const erl, ft const ech,
			     ft const ecl, bool const normalize)const
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


  // --- construct the profile with the angle integral --- //
  
  for(int n=0; n<NRAYS_LR; ++n){
    for(int m=0; m<NRAYS_HR; ++m){
      
      ft const ibetah = n_betah(n,m);
      if(ibetah < ft(1.e-14))
	continue;
      
      for(int ww=0; ww<N1; ++ww){
	
	ft const tr_hr = ft(1) / (ft(1) + fhr * sin2p_hr(m,ww));
	ft const tr_lr = ft(1) / (ft(1) + flr * sin2p_lr(n,ww));
	
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

void fpi::FPI::dual_fpi_full_der(int const N1, const ft* const tw, ft* const tr,
				 ft* const dtr, ft const erh, ft const erl, ft const ech,
				 ft const ecl, bool const normalize)const
{

  std::memset(tr,0,N1*sizeof(ft));
  std::memset(dtr,0,4*N1*sizeof(ft));

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
	ft const dtr_hr_dech = mth::SQ(tr_hr) * fhr * dsin2p_hr(m,ww);

	ft const dtr_lr_dech = dtr_lr_decl * decl_ech;


	dtr_derh[ww] += dtr_hr_derh * tr_lr * ibeta;
	dtr_derl[ww] += dtr_lr_derl * tr_hr * ibeta;

	dtr_decl[ww] += (dtr_lr_decl*tr_hr)*ibeta;
	dtr_dech[ww] += (dtr_hr_dech*tr_lr + tr_hr*dtr_lr_dech)*ibeta;

	tr[ww] += tr_hr * tr_lr * ibeta;
      }
    }
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

ft fpi::FPI::get_HRE_reflectivity()const
{
  return hr;
}

// ********************************************************************* //

ft fpi::FPI::get_LRE_reflectivity()const
{
  return lr;
}

// ********************************************************************* //
