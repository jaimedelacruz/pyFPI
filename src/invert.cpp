/* ---
   C++ wrappers to perform the inversion in parallel using OpenMP 

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
   --- */

#include <vector>
#include <omp.h>
#include <cstdio>
#include <array>

#include "invert.hpp"
#include "myTypes.hpp"
#include "math.hpp"
#include "lm.hpp"
#include "lm_containers.hpp"
#include "lm_containers_laser.hpp"

// **************************************************************************************** //

// --- parameter limits and scaling --- //
// --- in this order: scaling factor HRE, ECH, ERH, PR_CW,
//                    PR_FWHM, PR_NCAV, PR_LIN1, PR_LIN2, PR_LIN3,
//                    scaling factor LRE, ECL, ERL --- //

template<typename T>
constexpr inline static const std::array<T,12>
pscl = {1.0, 50., 0.01, 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 1.0, 50., 0.01};

template<typename T>
constexpr inline static const std::array<T,12>
pmax = {100.0, 300., 0.05, 1.0, 9.0, 3.0, 0.6, 0.2, 0.2, 10000, 300., 0.05};

template<typename T>
constexpr inline static const std::array<T,12>
pmin = {1.e-10, -300., -0.05, -1.0, 1.0, 1.4, -0.6, -0.2, -0.2, 1.e-5, -300., -0.05};



// **************************************************************************************** //

void fpi::invert_hre_crisp(long const ny, long const nx, long const npar, long const nwav, long const nfts, \
			   const ft* const fts_x, const ft* const fts_y, const float* const d, ft* const par, \
			   float* const syn, const ft* const wav,		\
			   std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw,
			   const int* const fixed, int const fpi_method, bool const no_pref, const float* const ecl,
			   const float* const erl)
{
  constexpr const long npar_fixed = 9;
  constexpr const int max_iter = 30;
  constexpr const ft chi_lim = 0.0;
  constexpr const int delay_bracket = 4;
  constexpr const ft I_THRES = 8.e-2;
  constexpr const ft init_lambda = 100.;

  
  // --- the parameters must be gain fractor, ech, erh, pr_w0, pr_fwhm, asym --- //

  if(npar != npar_fixed){
    fprintf(stderr,"[error] fpi::invert_hre_crisp: the number of parameters must be %ld but found %ld\n", npar_fixed, npar);
    exit(1);
  }
  
  
  // --- Initialize Levenberg-Marquardt class --- //
  
  int const nthreads = fpis.size();
  std::vector<std::unique_ptr<lm::LevMar<ft,float>>> inverters;

  for(int ii=0; ii<nthreads; ++ii){
    inverters.emplace_back(std::make_unique<lm::LevMar<ft,float>>(npar_fixed));
    for(int jj=0; jj<npar_fixed; ++jj){
      inverters[ii]->Pinfo[jj] = lm::Par<ft>(false, true, ((fixed[jj] == 0) ? false : true),\
					     pscl<ft>[jj], pmin<ft>[jj], pmax<ft>[jj]);
    }
  }

  
  // --- init parallel block and process data --- //
  
  long const npix = nx*ny;
  ft const scl = 100. / std::max<double>(double(npix-1),1.0);
  int odir = -1, dir=0, tid=0;
  long ipix =0;

  fprintf(stderr,"[info] invert_hre_crisp: processing -> %4d%%", dir);
  
  
#pragma omp parallel default(shared) firstprivate(tid,odir,dir, ipix) num_threads(nthreads)  
    {
      // --- get thread number --- //


      tid = omp_get_thread_num();


      // --- Init data container for HRE inversion --- //
      
      lm::container_base<ft,float>* const myData =
	new lm::container_hre_fit<ft,float>(nwav, *fpis[tid], d, sig, inverters[tid]->Pinfo,
					    nfts, fts_x, fts_y, wav, tw, fpi_method, no_pref, 0.0, 0.0);
      

#pragma omp for schedule(dynamic,20)
      for( ipix = 0; ipix<npix; ++ipix){

	// --- Assign pixel data to this thread --- //
	
	myData->d = d + ipix*nwav;

	((lm::container_hre_fit<ft,float>*)myData)->ecl = ft(ecl[ipix]);
	((lm::container_hre_fit<ft,float>*)myData)->erl = ft(erl[ipix]);

	
	// --- calculate the mean intensity of the pixel --- //
	
	float imax = ft(0);
	for(int ii=0; ii<nwav; ++ii)
	  imax= std::max(myData->d[ii], imax);
	
	

	// --- only perform the inversion if the mean intensity is significant,
	//     otherwise we assume that we are seeing the field stopper --- //
       
	if(imax > I_THRES){
	    
	
	  inverters[tid]->fitData(*myData, nwav, syn+ipix*nwav, par+ipix*npar, max_iter,
				  init_lambda, chi_lim,  ft(1.e-3), delay_bracket, false);
	  
	}else{
	  
	  // --- Else zero the gain so we can easily create a mask later --- //
	  
	  par[ipix*npar] = ft(0);
	  
	}


	// --- If tid == 0 printout some info, broken with clang in OSX? --- //
	
	if(tid == 0){
	  dir = std::round(scl*ft(ipix));
	  if(dir != odir){
	    odir = dir;
	    fprintf(stderr,"\r[info] invert_hre_crisp: processing -> %4d%%", dir);
	  }
	} // tid == 0
	
      } // ipix


      // --- Cleanup data struct used in the fits --- //
      
      delete myData;
      
    } // parallel block
 
    fprintf(stderr,"\r[info] invert_hre_crisp: processing -> %4d%%\n",100);

  
}

// **************************************************************************************** //


void fpi::invert_lre_crisp(long const ny, long const nx, long const npar, long const nwav, \
			   const float* const d, ft* const par, float* const syn, const ft* const wav, \
			   std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw, \
			   const float* const pref, const ft* const ech, const ft* const erh,
			   int const fpi_method)
{
  constexpr const long npar_fixed = 3;
  constexpr const int max_iter = 30;
  constexpr const ft chi_lim = 0.0;
  constexpr const int delay_bracket = 2;
  constexpr const ft I_THRES = 8.e-2;
  constexpr const ft init_lambda = 100.;
  
  // --- Initialize Levenberg-Marquardt class --- //
  
  int const nthreads = fpis.size();
  std::vector<std::unique_ptr<lm::LevMar<ft,float>>> inverters;

  for(int ii=0; ii<nthreads; ++ii){
    inverters.emplace_back(std::make_unique<lm::LevMar<ft,float>>(npar_fixed));
    for(int jj=0; jj<npar_fixed; ++jj)
      inverters[ii]->Pinfo[jj] = lm::Par<ft>(false, true, false, pscl<ft>[jj+9], pmin<ft>[jj+9], pmax<ft>[jj+9]);
  }


  // --- init parallel block and process data --- //
  
  long const npix = nx*ny;
  ft scl = 100. / std::max<double>(double(npix-1),1.0);
  int odir = -1, dir=0;


  fprintf(stderr,"[info] invert_lre_crisp: processing -> %4d%%", dir);

#pragma omp parallel default(shared) num_threads(nthreads)  
  {

      int const tid = omp_get_thread_num();
      
      lm::container_lre_fit<ft,float>* const myData =
	new lm::container_lre_fit<ft,float>(nwav, wav, d, sig, inverters[tid]->Pinfo,
					    *fpis[tid], tw, ech[0], erh[0], pref, fpi_method);
      
      
#pragma omp for schedule(dynamic,20)
      for(long ipix = 0; ipix<npix; ++ipix){
	
	myData->d    = d    + ipix*nwav;
	myData->pref = pref + ipix*nwav;
	
	myData->ech = ech[ipix];
	myData->erh = erh[ipix];


	float imax = ft(0);
	for(int ii=0;ii<nwav; ++ii)
	  imax = std::max(myData->d[ii], imax);

	if(imax >= I_THRES){
	
	
	  inverters[tid]->fitData( *((lm::container_base<ft,float>*)myData), nwav, syn+ipix*nwav, par+ipix*npar, max_iter,
				   init_lambda, chi_lim,  ft(1.e-3), delay_bracket, false);
	}else{
	  
	  // --- Else zero the gain so we can easily create a mask later --- //
	  
	  par[ipix*npar] = ft(0);
	  
	}

	if(tid == 0){
	  dir = std::round(scl*ipix);
	  if(dir != odir){
	    odir = dir;
	    fprintf(stderr,"\r[info] invert_lre_crisp: processing -> %4d%%", dir);
	  }
	}
      }

      delete myData;
      
    } // parallel block

  fprintf(stderr,"\r[info] invert_lre_crisp: processing -> %4d%%\n",100);

  
}

// **************************************************************************************** //

void fpi::invert_lre_crisp_laser(long const ny, long const nx, long const npar, long const nwav, \
				 const float* const d, ft* const par, float* const syn, const ft* const wav, \
				 std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw )
{
  constexpr const long npar_fixed = 3;
  constexpr const int max_iter = 20;
  constexpr const ft chi_lim = 0.0;
  constexpr const int delay_bracket = 2;
  constexpr const ft IMEAN_THRES = 1.e-2;

  
  // --- Initialize Levenberg-Marquardt class --- //
  
  int const nthreads = fpis.size();
  std::vector<std::unique_ptr<lm::LevMar<ft,float>>> inverters;

  for(int ii=0; ii<nthreads; ++ii){
    inverters.emplace_back(std::make_unique<lm::LevMar<ft,float>>(npar_fixed));
    for(int jj=0; jj<npar_fixed; ++jj)
      inverters[ii]->Pinfo[jj] = lm::Par<ft>(false, true, false, pscl<ft>[jj+8], pmin<ft>[jj+8], pmax<ft>[jj+8]);
  }


  // --- init parallel block and process data --- //
  
  long const npix = nx*ny;
  ft scl = 100. / std::max<double>(double(npix-1),1.0);
  int odir = -1, dir=0;
#pragma omp parallel default(shared) num_threads(nthreads)  
  {

      int const tid = omp_get_thread_num();
      
      lm::container_lre_laser_fit<ft,float>* const myData =
	new lm::container_lre_laser_fit<ft,float>(nwav, wav, d, sig, inverters[tid]->Pinfo, *fpis[tid], tw);
      
      
      
#pragma omp for schedule(dynamic,nx)
      for(long ipix = 0; ipix<npix; ++ipix){

	myData->d = d + ipix*nwav;

	ft imean = ft(0);
	for(int ii=0;ii<nwav; ++ii)
	  imean += myData->d[ii];
	imean /= nwav;

	if(imean >= IMEAN_THRES){
	
	  inverters[tid]->fitData( *((lm::container_base<ft,float>*)myData), nwav, syn+ipix*nwav, par+ipix*npar, max_iter,
				   ft(10), chi_lim,  ft(2.e-3), delay_bracket, false);
	}

	if(tid == 0){
	  dir = std::round(scl*ipix);
	  if(dir != odir){
	    odir = dir;
	    fprintf(stderr,"\r[info] invert_lre_crisp_laser: processing -> %4d%%", dir);
	  }
	}
      }

      delete myData;
      
    } // parallel block

  fprintf(stderr,"\n[info] invert_lre_crisp_laser: processing -> %4d%%\n",100);



}

// **************************************************************************************** //


void fpi::invert_hre_crisp_laser(long const ny, long const nx, long const npar, long const nwav, \
				 const float* const d, ft* const par, float* const syn, const ft* const wav, \
				 std::vector<fpi::FPI*> &fpis, const ft* const sig, const ft* const tw )
{
  constexpr const long npar_fixed = 3;
  constexpr const int max_iter = 20;
  constexpr const ft chi_lim = 0.0;
  constexpr const int delay_bracket = 2;
  constexpr const ft IMEAN_THRES = 1.e-2;
    
  
  // --- Initialize Levenberg-Marquardt class --- //
  
  int const nthreads = fpis.size();
  std::vector<std::unique_ptr<lm::LevMar<ft,float>>> inverters;

  for(int ii=0; ii<nthreads; ++ii){
    inverters.emplace_back(std::make_unique<lm::LevMar<ft,float>>(npar_fixed));
    for(int jj=0; jj<npar_fixed; ++jj)
      inverters[ii]->Pinfo[jj] = lm::Par<ft>(false, true, false, pscl<ft>[jj+8], pmin<ft>[jj+8], pmax<ft>[jj+8]);
  }


  // --- init parallel block and process data --- //
  
  long const npix = nx*ny;
  ft scl = 100. / std::max<double>(double(npix-1),1.0);
  int odir = -1, dir=0;

  
#pragma omp parallel default(shared) num_threads(nthreads)  
  {

      int const tid = omp_get_thread_num();
      
      lm::container_hre_laser_fit<ft,float>* const myData =
	new lm::container_hre_laser_fit<ft,float>(nwav, wav, d, sig, inverters[tid]->Pinfo, *fpis[tid], tw);
      
      
      
#pragma omp for schedule(dynamic,nx)
      for(long ipix = 0; ipix<npix; ++ipix){
	
	myData->d = d + ipix*nwav;

	ft imean = ft(0);
	for(int ii=0;ii<nwav; ++ii)
	  imean += myData->d[ii];
	imean /= nwav;
	
	if(imean >= IMEAN_THRES){
	
	  inverters[tid]->fitData( *((lm::container_base<ft,float>*)myData), nwav, syn+ipix*nwav, par+ipix*npar, max_iter,
				   ft(10), chi_lim,  ft(2.e-3), delay_bracket, false);
	  
	}
	if(tid == 0){
	  dir = std::round(scl*ipix);
	  if(dir != odir){
	    odir = dir;
	    fprintf(stderr,"\r[info] invert_hre_crisp_laser: processing -> %4d%%", dir);
	  }
	}
      }

      delete myData;
      
    } // parallel block

  fprintf(stderr,"\n[info] invert_hre_crisp_laser: processing -> %4d%%\n",100);



}

// **************************************************************************************** //

void fpi::invert_all_crisp(long const ny, long const nx, long const npar, long const nwavh, long const nwavl, long const nfts, \
			   const ft* const fts_x, const ft* const fts_y, const ft* const fts_yl, const float* const dh, const float* const dl,
			   ft* const par, float* const syn, const ft* const wavh, const ft* const wavl, \
			   std::vector<fpi::FPI*> &fpis, const ft* const sigh, const ft* const sigl, const ft* const tw, const int* const fixed,
			   int const fpi_method, ft const dwgrid)
{

  constexpr const long npar_fixed = 12;
  constexpr const int max_iter = 35;
  constexpr const ft chi_lim = 0.0;
  constexpr const int delay_bracket = 4;
  constexpr const ft IMEAN_THRES = 7.e-2;
  constexpr const ft init_lambda = 31.622776601683795;
  long const nwav = nwavl + nwavh;
  
  
  // --- the parameters must be gain fractor, ech, erh, pr_w0, pr_fwhm, asym --- //

  if(npar != npar_fixed){
    fprintf(stderr,"[error] fpi::invert_all_crisp: the number of parameters must be %ld but found %ld\n", npar_fixed, npar);
    exit(1);
  }
  

  fprintf(stderr,"[info] fpi::invert_all_crisp: nwavh=%ld, nwavl=%ld, nfts=%ld\n",nwavh, nwavl,nfts);
  
  // --- Initialize Levenberg-Marquardt class --- //
  
  int const nthreads = fpis.size();
  std::vector<std::unique_ptr<lm::LevMar<ft,float>>> inverters;

  for(int ii=0; ii<nthreads; ++ii){
    inverters.emplace_back(std::make_unique<lm::LevMar<ft,float>>(npar_fixed));
    for(int jj=0; jj<npar_fixed; ++jj){
      inverters[ii]->Pinfo[jj] = lm::Par<ft>(false, true, ((fixed[jj] == 0) ? false : true),\
					     pscl<ft>[jj], pmin<ft>[jj], pmax<ft>[jj]);
    }
  }

  
  // --- init parallel block and process data --- //
  
  long const npix = nx*ny;
  ft const scl = 100. / std::max<double>(double(npix-1),1.0);
  int odir = -1, dir=0, tid=0;
  long ipix =0;

  fprintf(stderr,"[info] invert_all_crisp: processing -> %4d%%", dir);

  
#pragma omp parallel default(shared) firstprivate(tid,odir,dir, ipix) num_threads(nthreads)  
    {
      // --- get thread number --- //


      tid = omp_get_thread_num();


      // --- Init data container for HRE inversion --- //
	    
      lm::container_base<ft,float>* const myData =
	new lm::container_all_fit<ft,float>(nwavh, nwavl, *fpis[tid], dh, dl, sigh, sigl, inverters[tid]->Pinfo,
					    nfts, fts_x, fts_y, fts_yl, wavh, wavl, tw, fpi_method, dwgrid);
      

#pragma omp for schedule(dynamic,100)
      for( ipix = 0; ipix<npix; ++ipix){

	// --- Assign pixel data to this thread --- //
	
	myData->d = dh + ipix*nwavh;
	((lm::container_all_fit<ft,float>*)myData)->dl = dl+ipix*nwavl;
	
	
	// --- calculate the mean intensity of the pixel --- //
	
	ft imean = ft(0);
	for(int ii=0; ii<nwavh; ++ii)
	  imean += myData->d[ii];
	imean /= nwav;

	

	// --- only perform the inversion if the mean intensity is significant,
	//     otherwise we assume that we are seeing the field stopper --- //
       
	if(imean > IMEAN_THRES){
	    
	
	  inverters[tid]->fitData(*myData, nwav, syn+ipix*nwav, par+ipix*npar, max_iter,
				  init_lambda, chi_lim,  ft(1.e-3), delay_bracket, false);
	  
	}else{
	  
	  // --- Else zero the gain so we can easily create a mask later --- //
	  
	  par[ipix*npar] = ft(0);
	  par[ipix*npar+9] = ft(0);
	  
	}


	// --- If tid == 0 printout some info, broken with clang in OSX? --- //
	
	if(tid == 0){
	  dir = std::round(scl*ft(ipix));
	  if(dir != odir){
	    odir = dir;
	    fprintf(stderr,"\r[info] invert_all_crisp: processing -> %4d%%", dir);
	  }
	} // tid == 0
	
      } // ipix


      // --- Cleanup data struct used in the fits --- //
      
      delete myData;
      
    } // parallel block
 
    fprintf(stderr,"\r[info] invert_all_crisp: processing -> %4d%%\n",100);

  

}

// **************************************************************************************** //
