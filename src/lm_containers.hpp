#ifndef LMCONTHPP
#define LMCONTHPP

/* ---

   Inherited subclasses implementing the model calculation for each case
   Coded by J. de la Cruz Rodriguez (ISP-SU 2025)
   
   --- */

#include "math.hpp"
#include "fpi.hpp"
#include "lm.hpp"
#include "prefilter.hpp"

namespace lm{

  // ***************************************** //

  template<typename T, typename U>
  struct container_lre_fit: public container_base<T,U>{
    
    int Nreal;
    fpi::FPI const& ifpi;
    const T* const tw;
    T ech;
    T erh;
    const U* pref;
    
    // --- temporary variables for calculations in fx and fx_dx --- //
    T* const m;
    T* const ltr;
    T* const dtr;

    int const fpi_method;
    
    // ----------------------------------------------------------------- //

    ~container_lre_fit()
    {
      delete [] m;
      delete [] ltr;
      delete [] dtr;
    }
    
    // ----------------------------------------------------------------- //

    inline container_lre_fit(int const nd, const T* const iwav, const U* const din, const T* const sigin, \
			     std::vector<Par<T>> const& Pi,fpi::FPI const& ifpi_in, const T* const itw,
			     T const echin, T const erhin, const U* const ipref, int const ifpi_method):
      container_base<T,U>(nd,iwav,din,sigin,Pi),
      Nreal(nd), ifpi(ifpi_in), tw(itw), ech(echin), erh(erhin),pref(ipref),
      //
      m(new T[Pi.size()]()),
      ltr(new T[ifpi.convolver->n1]()),
      dtr(new T[5*ifpi.convolver->n1]()),
      fpi_method(ifpi_method){};

    // ----------------------------------------------------------------- //
    
    T fx(int const nPar, const T* __restrict__ m_in,
	 T* const __restrict__ syn, T* const __restrict__ r)const
    {
      
      constexpr T const angle=T(0);
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      
      
      // --- Copy model --- //
      
      std::memcpy(m,m_in,nPar*sizeof(T));

      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles --- //

      if(fpi_method == 0){
	ifpi.dual_fpi_ray_individual(npsf, tw, syn, ltr, erh, m[2], ech, m[1], angle, true, false);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv_individual(npsf, tw, syn, ltr, erh, m[2], ech, m[1], true, false);
      }else{
	ifpi.dual_fpi_full_individual(npsf, tw, syn, ltr, erh, m[2], ech, m[1], true, false);
      }

      for(int ii=0; ii<npsf; ++ii){
	syn[ii] *= m[0] * pref[ii];
      }
      
      
      ifpi.convolver->updatePSF(npsf, ltr);
      ifpi.convolver->convolve(npsf, syn);

      
      
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDat; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }


      return this->getChi2(nDat, r);
    }
    
    // ----------------------------------------------------------------- //
    
    T fx_dx(int const nPar, const T* const __restrict__ m_in,
	    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const
    {
      constexpr T const angle=T(0);
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      
      
      // --- Copy model --- //

      std::memcpy(m,m_in,nPar*sizeof(T));

      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles and derivatives --- //

      if(fpi_method == 0){
	ifpi.dual_fpi_ray_individual_der(npsf, tw, syn, ltr, dtr, erh, m[2], ech, m[1], angle, true, false);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv_individual_der(npsf, tw, syn, ltr, dtr, erh, m[2], ech, m[1], true, false);
      }else{
	ifpi.dual_fpi_full_individual_der(npsf, tw, syn, ltr, dtr, erh, m[2], ech, m[1], true, false);
      }

      T* const __restrict__ dtr_decl =  dtr+npsf*3;
      T* const __restrict__ dtr_derl =  dtr+npsf;
      
      
      for(int ii=0; ii<npsf; ++ii){
	//T const iscl = pref[ii]*m[0];
      	syn[ii] *= pref[ii];
	J[nDat*2+ii] = J[nDat+ii] = syn[ii]*m[0];
      }

      ifpi.convolver->updatePSF(npsf,ltr);
      ifpi.convolver->convolve(npsf, syn);

      ifpi.convolver->updatePSF(npsf,dtr_decl);
      ifpi.convolver->convolve(npsf, J+nDat);
      
      ifpi.convolver->updatePSF(npsf,dtr_derl);
      ifpi.convolver->convolve(npsf, J+2*nDat);

      
      for(int ii=0; ii<nDat; ++ii){
	J[ii] = syn[ii];
	syn[ii] *= m[0];	
      }

      //std::memcpy(J+nDat,dtr+npsf*3, nDat*sizeof(T));
      //std::memcpy(J+nDat*2,dtr+npsf, nDat*sizeof(T));
      
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDat; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }

      
            
      // --- scale J --- //
      
      for(int ii = 0; ii<nPar; ++ii){
	
	T const iScl = this->Pinfo[ii].scale / scl;
	T* const __restrict__ iJ = J + ii*nDat;
	
	for(int ww = 0; ww<nDat; ++ww)
	  iJ[ww] *=  iScl / sig[ww];
      }
      
      
      return this->getChi2(nDat, r);
    }
  };

  
  // ***************************************** //

  template<typename T,typename U>
  struct container_hre_fit: public container_base<T,U>{
    
    int Nreal;
    fpi::FPI const& ifpi;

    int nfts;
    const T* const fts_x;
    const T* const fts_y;
    const T* const tw;
    int const fpi_method;
    bool const no_pref;
    T ecl;
    T erl;
    
    // --- internal variables used in fx and fx_dx --- //
    T* const m;
    T* const tr;
    T* const dtr;
    T* const cfts_y;
    T* const dfts_y;

    // ------------------------------------------- //

    ~container_hre_fit()
    {
      // --- clean-up allocated memory --- //
      
      delete [] m;
      delete [] tr;
      delete [] dtr;
      delete [] cfts_y;
      delete [] dfts_y;
    }
    
    // ------------------------------------------- //

    container_hre_fit(int const nd, fpi::FPI const& ifpi_in, const U* const __restrict__ din, \
		      const T* const __restrict__ sigin, const std::vector<Par<T>> &Pi, \
		      int const infts, const T* const ifts_x, const T* const ifts_y, \
		      const T* const iwav, const T* const itw, int const ifpi_method, bool const npref,
		      T const iecl, T const ierl):
      container_base<T,U>(nd,iwav,din,sigin,Pi),
      Nreal(1), ifpi(ifpi_in), nfts(infts), fts_x(ifts_x),
      fts_y(ifts_y), tw(itw), fpi_method(ifpi_method), no_pref(npref),ecl(iecl), erl(ierl),
      // --- allocate internal buffers --- //
      m(new T[Pi.size()]()),
      tr(new T[nfts]()),
      dtr(new T[5*nfts]()),
      cfts_y(new T[nfts]()),
      dfts_y(new T[8*nfts]())
    {
      // --- only account for non-dummy points in the data array --- //
      
      Nreal = 0;
      for(int ii = 0; ii<this->nDat; ++ii)
	if(this->sig[ii] < 1.e20) Nreal += 1;
      
    }

    // ------------------------------------------- //

    T fx(int const nPar, const T* __restrict__ m_in,
	 T* const __restrict__ syn, T* const __restrict__ r)const
    {
      bool normalize_tr;
      constexpr ft const angle = 0;
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      
      
      // --- Copy model --- //

      std::memcpy(m,m_in,nPar*sizeof(T));
      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      const T* const __restrict__ wav = this->x;      
      
      
      // --- apply prefilter to fts --- //

      if(no_pref){
	for(int ii=0; ii<nfts; ++ii){
	  T const lin = 1.0 + (m[6] + (m[7] + m[8]*fts_x[ii])*fts_x[ii])*fts_x[ii];	  
	  cfts_y[ii] = fts_y[ii] * lin * m[0];
	}
      }else{
	
	for(int ii=0; ii<nfts; ++ii){
	  cfts_y[ii] = pref::prefilter(fts_x[ii],m[0],m[3],m[4],m[5],m[6],m[7],m[8],fts_y[ii]);
	}
	
      }
      
      // --- calculate transmission profile --- //
      if(fpi_method == 0){
	ifpi.dual_fpi_ray(npsf, tw, tr, m[2], erl, m[1], ecl, angle, normalize_tr = true);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv(npsf, tw, tr, m[2], erl, m[1], ecl, normalize_tr = true);
      }else{
	ifpi.dual_fpi_full(npsf, tw, tr, m[2], erl, m[1], ecl, normalize_tr = true);
      }
      
      ifpi.convolver->updatePSF(npsf, tr);
      
      

      // --- convolve fts atlas --- //

      ifpi.convolver->convolve(nfts,cfts_y);
	

      // --- now we will need to interpolate to the observed grid --- //

      mth::interpolation_Linear<T>(nfts, fts_x, cfts_y, nDat, wav, syn);

    
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDat; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }

      
      // --- get Chi2 --- //
      
      return this->getChi2(nDat, r);  
    }
    
    // ******************************************* //
    
    T fx_dx(int const nPar, const T* const __restrict__ m_in,
	    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const
    {
      bool normalize_tr;
      constexpr ft const angle = ft(0);
      
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      T dp[7] = {};
      
      // --- Copy model --- //

      std::memcpy(m,m_in,nPar*sizeof(T));
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      const T* const __restrict__ wav = this->x;
      
      T* const __restrict__ dfts_dech = dfts_y + 0*nfts;
      T* const __restrict__ dfts_derh = dfts_y + 1*nfts;
      T* const __restrict__ dfts_dpcw = dfts_y + 2*nfts;
      T* const __restrict__ dfts_dpfw = dfts_y + 3*nfts;
      T* const __restrict__ dfts_dpex = dfts_y + 4*nfts;
      T* const __restrict__ dfts_dli1 = dfts_y + 5*nfts;
      T* const __restrict__ dfts_dli2 = dfts_y + 6*nfts;
      T* const __restrict__ dfts_dli3 = dfts_y + 7*nfts;

      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      
      // --- apply prefilter to fts --- //

      if(no_pref){
	for(int ii=0; ii<nfts; ++ii){
	  T const x2 = mth::SQ(fts_x[ii]);
	  T const iFTS = fts_y[ii] * m[0];
	  T const lin = 1.0 + (m[6] + (m[7] + m[8]*fts_x[ii])*fts_x[ii])*fts_x[ii];	  

	  cfts_y[ii]    = iFTS  * lin;
	  dfts_dli1[ii] = iFTS  * fts_x[ii];
	  dfts_dli2[ii] = iFTS  * x2;
	  dfts_dli3[ii] = iFTS  * x2*fts_x[ii];
	  
	  
	}
      }else{
	
	for(int ii=0; ii<nfts; ++ii){

	  // --- calculate prefilter curve and the Jacobian -> dp  --- //
	  cfts_y[ii] = pref::Dprefilter(fts_x[ii],m[0],m[3],m[4],m[5],m[6],m[7],m[8],fts_y[ii],dp);
	  
	  dfts_dpcw[ii] = dp[1];
	  dfts_dpfw[ii] = dp[2];
	  dfts_dpex[ii] = dp[3];
	  dfts_dli1[ii] = dp[4];
	  dfts_dli2[ii] = dp[5];
	  dfts_dli3[ii] = dp[6];
	}

      }

      
      // --- make a copy of the prefilter * fts --- //
      
       if(!this->Pinfo[2].fixed) std::memcpy(dfts_derh, cfts_y, sizeof(T)*nfts);
       if(!this->Pinfo[1].fixed) std::memcpy(dfts_dech, cfts_y, sizeof(T)*nfts);
      
      
      // --- calculate transmission profile --- //

       if(fpi_method == 0){
	 ifpi.dual_fpi_ray_der(npsf, tw, tr, dtr, m[2], erl, m[1], ecl, angle, normalize_tr = true);
       }else if(fpi_method == 1){
	 ifpi.dual_fpi_conv_der(npsf, tw, tr, dtr, m[2], erl, m[1], ecl, normalize_tr = true);
       }else{
	 ifpi.dual_fpi_full_der(npsf, tw, tr, dtr, m[2], erl, m[1], ecl, normalize_tr = true);
       }
      
      // --- now propagate the derivatives of the instrumental profile through the convolution --- //

      if(!this->Pinfo[2].fixed){
	ifpi.convolver->updatePSF(npsf,dtr);
	ifpi.convolver->convolve(nfts, dfts_derh);
      }

      if(!this->Pinfo[1].fixed){
	ifpi.convolver->updatePSF(npsf,dtr+2*npsf);
	ifpi.convolver->convolve(nfts, dfts_dech);
      }
      
      
      // --- convolve fts atlas and all quantities that do not change the instrumental profile --
      
      ifpi.convolver->updatePSF(npsf,tr);
      ifpi.convolver->convolve(nfts,cfts_y);
      if(!this->Pinfo[3].fixed && !no_pref) ifpi.convolver->convolve(nfts,dfts_dpcw);
      if(!this->Pinfo[4].fixed && !no_pref) ifpi.convolver->convolve(nfts,dfts_dpfw);
      if(!this->Pinfo[5].fixed && !no_pref) ifpi.convolver->convolve(nfts,dfts_dpex);
      if(!this->Pinfo[6].fixed) ifpi.convolver->convolve(nfts,dfts_dli1);
      if(!this->Pinfo[7].fixed) ifpi.convolver->convolve(nfts,dfts_dli2);
      if(!this->Pinfo[8].fixed) ifpi.convolver->convolve(nfts,dfts_dli3);
  
      
      // --- interpolate everything to the observed grid --- //
      
      mth::interpolation_Linear<T>(nfts, fts_x, cfts_y,    nDat, wav, syn);
      if(!this->Pinfo[1].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dech, nDat, wav, J+1*nDat);
      if(!this->Pinfo[2].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_derh, nDat, wav, J+2*nDat);
      if(!this->Pinfo[3].fixed && !no_pref) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpcw, nDat, wav, J+3*nDat);
      if(!this->Pinfo[4].fixed && !no_pref) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpfw, nDat, wav, J+4*nDat);
      if(!this->Pinfo[5].fixed && !no_pref) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpex, nDat, wav, J+5*nDat);
      if(!this->Pinfo[6].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli1, nDat, wav, J+6*nDat);
      if(!this->Pinfo[7].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli2, nDat, wav, J+7*nDat);
      if(!this->Pinfo[8].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli3, nDat, wav, J+8*nDat);


      if(!this->Pinfo[0].fixed){
	
	// --- copy the unscaled prefilter + fts to the derivative relative to m[0] --- //
	
	std::memcpy(J,syn,nDat*sizeof(T));
	
	
	// --- now apply the scale factor ---- //
	
	for(int ii=0; ii<nDat; ++ii){
	  J[ii] /= m[0];
	}
      }
      
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal)); 
      for(int ii=0; ii<nDat; ++ii)
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      

      
      // --- scale J --- //
      
      for(int ii = 0; ii<nPar; ++ii){
	if(!this->Pinfo[ii].fixed){
	  T const iScl = this->Pinfo[ii].scale / scl;
	  T* const __restrict__ iJ = J + ii*nDat;
	  
	  for(int ww = 0; ww<nDat; ++ww)
	    iJ[ww] *=  iScl / sig[ww];
	}
      }
      

      // --- get Chi2 --- //
      
      return this->getChi2(nDat, r);  
    }
    
    
    // ***************************************** //
    
    
  };
  
  
   // ***************************************** //
  
  template<typename T,typename U>
  struct container_all_fit: public container_base<T,U>{
    
    int Nreal;
    fpi::FPI const& ifpi;

    int nfts;
    const T* const fts_x;
    const T* const fts_y;
    const T* const tw;
    const T* const fts_yl;
    const U* dl;
    const T* const wavl;
    const T* const sigl;
    T const dwgrid;
    int const fpi_method;
    int const nDatH;
    int const nDatL;
    
    
    // --- internal variables used in fx and fx_dx --- //
    
    T* const m;
    T* const tr;
    T* const dtr;
    T* const cfts_y;
    T* const dfts_y;
    T* const ltr;
    T* const htr;
    T* const dtr1;
    T* const wavl_dwgrid;
    
    // ------------------------------------------- //

    ~container_all_fit()
    {
      // --- clean-up allocated memory --- //
      
      delete [] m;
      delete [] tr;
      delete [] dtr;
      delete [] cfts_y;
      delete [] dfts_y;
      delete [] ltr;
      delete [] htr;
      delete [] dtr1;
      delete [] wavl_dwgrid;
    }
    
    // ------------------------------------------- //
    //(nwavh, nwavl, *fpis[tid], dh, hl, sigh, sigl, inverters[tid]->Pinfo,
    //					    nfts, fts_x, fts_y, wavh, wavl, tw, fpi_method);
    
    container_all_fit(int const nd, int const ndl, fpi::FPI const& ifpi_in, const U* const __restrict__ din, \
		      const U* const __restrict__ dinl, 
		      const T* const __restrict__ sigin, const T* const __restrict__ siginl,  \
		      const std::vector<Par<T>> &Pi, 		\
		      int const infts, const T* const ifts_x, const T* const ifts_y, const T* const ifts_yl, \
		      const T* const iwav, const T* const iwavl, const T* const itw, int const ifpi_method, T const idwgrid):
      container_base<T,U>(nd+ndl,iwav,din,sigin,Pi),
      Nreal(1), ifpi(ifpi_in), nfts(infts), fts_x(ifts_x),
      fts_y(ifts_y), tw(itw), fts_yl(ifts_yl), dl(dinl), wavl(iwavl), sigl(siginl), dwgrid(idwgrid),
      fpi_method(ifpi_method), nDatH(nd), nDatL(ndl),
      
      // --- allocate internal buffers --- //
      m(new T[Pi.size()]()),
      tr(new T[nfts]()),
      dtr(new T[5*nfts]()),
      cfts_y(new T[nfts]()),
      dfts_y(new T[10*nfts]()),
      ltr(new T[ndl]()),
      htr(new T[ndl]()),
      dtr1(new T[11*ndl]()),
      wavl_dwgrid(new T[nDatL]())
    {
      // --- only account for non-dummy points in the data array --- //
      
      Nreal = 0;
      for(int ii = 0; ii<this->nDat; ++ii)
	if(this->sig[ii] < 1.e20) Nreal += 1;
      
      for(int ii = 0; ii<nDatL; ++ii)
	if(sigl[ii] < 1.e20) Nreal += 1;


      // --- precompute wavl_dwgrid --- //

      for(int ii=0; ii<nDatL; ++ii)
	wavl_dwgrid[ii] = wavl[ii] + dwgrid;
      
    }

    // ------------------------------------------- //

    T fx(int const nPar, const T* __restrict__ m_in,
	 T* const __restrict__ syn, T* const __restrict__ r)const
    {
      bool normalize_tr;
      constexpr ft const angle = 0;
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      
      
      // --- Copy model --- //

      std::memcpy(m,m_in,nPar*sizeof(T));

      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      const T* const __restrict__ wav = this->x;      
      
      
      // --- apply prefilter to fts --- //

      {

	for(int ii=0; ii<nfts; ++ii){
	  cfts_y[ii] = pref::prefilter(fts_x[ii],m[0],m[3],m[4],m[5],m[6],m[7],m[8],fts_y[ii]);
	}
      }
      
      // --- calculate transmission profile --- //
      
      if(fpi_method == 0){
	ifpi.dual_fpi_ray(npsf, tw, tr, m[2], m[11], m[1], m[10], angle, normalize_tr = true);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv(npsf, tw, tr, m[2], m[11], m[1], m[10], normalize_tr = true);
      }else{
	ifpi.dual_fpi_full(npsf, tw, tr, m[2], m[11], m[1], m[10], normalize_tr = true);
      }
      
      
      // --- convolve fts atlas --- //
      
      ifpi.convolver->updatePSF(npsf, tr);
      ifpi.convolver->convolve(nfts,cfts_y);
	

      // --- now we will need to interpolate to the observed grid --- //

      mth::interpolation_Linear<T>(nfts, fts_x, cfts_y, nDatH, wav, syn);

    
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDatH; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }


      // --- now the LRE model --- //

      
      // --- calculate transmission profiles --- //

      if(fpi_method == 0){
	ifpi.dual_fpi_ray_individual(nDatL, wavl, htr, ltr, m[2], m[11], m[1], m[10], angle, true, false);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv_individual(nDatL, wavl, htr, ltr, m[2], m[11], m[1], m[10], true, false);
      }else{
	ifpi.dual_fpi_full_individual(nDatL, wavl, htr, ltr,  m[2], m[11], m[1], m[10], true, false);
      }
      {
	
	for(int ii=0; ii<nDatL; ++ii){
	  syn[ii+nDatH] = pref::prefilter(wavl_dwgrid[ii],m[9],m[3],m[4],m[5],m[6],m[7],m[8],fts_yl[ii]) * htr[ii];
	}
      }

      
      ifpi.convolver2->updatePSF(nDatL, ltr);
      ifpi.convolver2->convolve(nDatL, syn+nDatH);

      
      for(int ii=0; ii<nDatL; ++ii){
	r[ii+nDatH] = (T(dl[ii]) - syn[ii+nDatH]) / (sigl[ii] * scl);
      }

      
      // --- get Chi2 --- //
      
      return this->getChi2(nDat, r);  
    }
    

    // ------------------------------------------- //

    T fx_dx(int const nPar, const T* const __restrict__ m_in,
	    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const
    {
      bool normalize_tr;
      constexpr ft const angle = ft(0);
      
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      T dp[7] = {};

      std::memset(J,0,nPar*nDat*sizeof(T));
      
      
      // --- Copy model --- //

      std::memcpy(m,m_in,nPar*sizeof(T));
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      const T* const __restrict__ wav = this->x;
      
      T* const __restrict__ dfts_dech = dfts_y + 0*nfts;
      T* const __restrict__ dfts_derh = dfts_y + 1*nfts;
      T* const __restrict__ dfts_dpcw = dfts_y + 2*nfts;
      T* const __restrict__ dfts_dpfw = dfts_y + 3*nfts;
      T* const __restrict__ dfts_dpex = dfts_y + 4*nfts;
      T* const __restrict__ dfts_dli1 = dfts_y + 5*nfts;
      T* const __restrict__ dfts_dli2 = dfts_y + 6*nfts;
      T* const __restrict__ dfts_dli3 = dfts_y + 7*nfts;
      T* const __restrict__ dfts_decl = dfts_y + 8*nfts;
      T* const __restrict__ dfts_derl = dfts_y + 9*nfts;

      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      
      // --- apply prefilter to fts --- //
      {
	
	for(int ii=0; ii<nfts; ++ii){
	  
	  // --- calculate prefilter curve and the Jacobian -> dp  --- //
	  cfts_y[ii] = pref::Dprefilter(fts_x[ii],m[0],m[3],m[4],m[5],m[6],m[7],m[8],fts_y[ii],dp);
	  
	  dfts_dpcw[ii] = dp[1];
	  dfts_dpfw[ii] = dp[2];
	  dfts_dpex[ii] = dp[3];
	  dfts_dli1[ii] = dp[4];
	  dfts_dli2[ii] = dp[5];
	  dfts_dli3[ii] = dp[6];
	}
	
      }
      
      // --- make a copy of the prefilter * fts --- //

      if(!this->Pinfo[1].fixed)  std::memcpy(dfts_dech, cfts_y, sizeof(T)*nfts);
      if(!this->Pinfo[2].fixed)  std::memcpy(dfts_derh, cfts_y, sizeof(T)*nfts);
      if(!this->Pinfo[10].fixed) std::memcpy(dfts_decl, cfts_y, sizeof(T)*nfts);
      if(!this->Pinfo[11].fixed) std::memcpy(dfts_derl, cfts_y, sizeof(T)*nfts);

       
      
      // --- calculate transmission profile --- //

       if(fpi_method == 0){
	 ifpi.dual_fpi_ray_der(npsf, tw, tr, dtr, m[2], m[11], m[1], m[10], angle, normalize_tr = true);
       }else if(fpi_method == 1){
	 ifpi.dual_fpi_conv_der(npsf, tw, tr, dtr, m[2], m[11], m[1], m[10], normalize_tr = true);
       }else{
	 ifpi.dual_fpi_full_der(npsf, tw, tr, dtr, m[2], m[11], m[1], m[10], normalize_tr = true);
       }
      
      // --- now propagate the derivatives of the instrumental profile through the convolution --- //

      if(!this->Pinfo[1].fixed){
	ifpi.convolver->updatePSF(npsf,dtr+2*npsf);
	ifpi.convolver->convolve(nfts, dfts_dech);
      }
       
      if(!this->Pinfo[2].fixed){
	ifpi.convolver->updatePSF(npsf,dtr);
	ifpi.convolver->convolve(nfts, dfts_derh);
      }
      
      if(!this->Pinfo[10].fixed){
	ifpi.convolver->updatePSF(npsf,dtr+3*npsf);
	ifpi.convolver->convolve(nfts, dfts_decl);
      }
      
      if(!this->Pinfo[11].fixed){
	ifpi.convolver->updatePSF(npsf,dtr+1*npsf);
	ifpi.convolver->convolve(nfts, dfts_derl);
      }
      
      // --- convolve fts atlas and all quantities that do not change the instrumental profile --
      
      ifpi.convolver->updatePSF(npsf,tr);
      ifpi.convolver->convolve(nfts,cfts_y);
      if(!this->Pinfo[3].fixed) ifpi.convolver->convolve(nfts,dfts_dpcw);
      if(!this->Pinfo[4].fixed) ifpi.convolver->convolve(nfts,dfts_dpfw);
      if(!this->Pinfo[5].fixed) ifpi.convolver->convolve(nfts,dfts_dpex);
      if(!this->Pinfo[6].fixed) ifpi.convolver->convolve(nfts,dfts_dli1);
      if(!this->Pinfo[7].fixed) ifpi.convolver->convolve(nfts,dfts_dli2);
      if(!this->Pinfo[8].fixed) ifpi.convolver->convolve(nfts,dfts_dli3);
  
      
      // --- interpolate everything to the observed grid --- //
      
      mth::interpolation_Linear<T>(nfts, fts_x, cfts_y, nDatH, wav, syn);
      if(!this->Pinfo[1].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dech, nDatH, wav, J+1*nDat);
      if(!this->Pinfo[2].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_derh, nDatH, wav, J+2*nDat);
      if(!this->Pinfo[3].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpcw, nDatH, wav, J+3*nDat);
      if(!this->Pinfo[4].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpfw, nDatH, wav, J+4*nDat);
      if(!this->Pinfo[5].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dpex, nDatH, wav, J+5*nDat);
      if(!this->Pinfo[6].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli1, nDatH, wav, J+6*nDat);
      if(!this->Pinfo[7].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli2, nDatH, wav, J+7*nDat);
      if(!this->Pinfo[8].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_dli3, nDatH, wav, J+8*nDat);
      if(!this->Pinfo[10].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_decl, nDatH, wav, J+10*nDat);
      if(!this->Pinfo[11].fixed) mth::interpolation_Linear<T>(nfts, fts_x, dfts_derl, nDatH, wav, J+11*nDat);


      if(!this->Pinfo[0].fixed){
	
	// --- now apply the scale factor ---- //
	
	for(int ii=0; ii<nDatH; ++ii){
	  J[ii] = syn[ii] / m[0];
	}
      }

      // --- LRE data model --- //

      // --- calculate the prefilter in the LRE grid --- //

      T* const dpref_dcw = J+nDat*3+nDatH;
      T* const dpref_dfw = J+nDat*4+nDatH;
      T* const dpref_dex = J+nDat*5+nDatH;
      T* const dpref_dp0 = J+nDat*6+nDatH;
      T* const dpref_dp1 = J+nDat*7+nDatH;
      T* const dpref_dp2 = J+nDat*8+nDatH;
      T* const dpref_dm1  = J+nDat*9+nDatH;
      T* const dltr_decl = J+nDat*10+nDatH;
      T* const dltr_derl = J+nDat*11+nDatH;
      T* const dhtr_dech = J+nDat*1+nDatH;
      T* const dhtr_derh = J+nDat*2+nDatH;
      T* const dltr_dech = dtr1+nDatL*4;
      T* const __restrict__ synl = syn + nDatH;

      {
	
	for(int ii=0; ii<nDatL; ++ii){
	  synl[ii] = pref::Dprefilter(wavl_dwgrid[ii],m[9],m[3],m[4],m[5],m[6],m[7],m[8],fts_yl[ii],dp);
	  dpref_dcw[ii] = dp[1];
	  dpref_dfw[ii] = dp[2];
	  dpref_dex[ii] = dp[3];
	  dpref_dp0[ii] = dp[4];
	  dpref_dp1[ii] = dp[5];
	  dpref_dp2[ii] = dp[6];
	}
	
      }
      
      // --- calculate transmission profiles and derivatives --- //

      if(fpi_method == 0){
	ifpi.dual_fpi_ray_individual_der(nDatL, wavl, htr, ltr, dtr1, m[2], m[11], m[1], m[10], angle, true, false);
      }else if(fpi_method == 1){
	ifpi.dual_fpi_conv_individual_der(nDatL, wavl, htr, ltr, dtr1,  m[2], m[11], m[1], m[10], true, false);
      }else{
	ifpi.dual_fpi_full_individual_der(nDatL, wavl, htr, ltr, dtr1,  m[2], m[11], m[1], m[10], true, false);
      }

      T* const __restrict__ dtr1_decl =  dtr1+nDatL*3;
      T* const __restrict__ dtr1_derl =  dtr1+nDatL;     
      T* const __restrict__ dtr1_dech =  dtr1+nDatL*2;
      T* const __restrict__ dtr1_derh =  dtr1;
 
      
      for(int ii=0; ii<nDatL; ++ii){

	dhtr_dech[ii] = synl[ii] * dtr1_dech[ii];
	dhtr_derh[ii] = synl[ii] * dtr1_derh[ii];	
	
      	synl[ii] *= htr[ii];
	dltr_decl[ii] = synl[ii];
	dltr_derl[ii] = synl[ii];
	dpref_dm1[ii] = synl[ii]; // temporarily storing dltr_dech
	
	dpref_dcw[ii] *= htr[ii];
	dpref_dfw[ii] *= htr[ii];
	dpref_dex[ii] *= htr[ii];
	dpref_dp0[ii] *= htr[ii];
	dpref_dp1[ii] *= htr[ii];
	dpref_dp2[ii] *= htr[ii];
      }

      // --- Convolve quantities that do not involve the derivatives of the transmission profiles --- //
      
      ifpi.convolver2->updatePSF(nDatL,ltr);
      ifpi.convolver2->convolve(nDatL, synl);
      ifpi.convolver2->convolve(nDatL, dpref_dcw);
      ifpi.convolver2->convolve(nDatL, dpref_dfw);
      ifpi.convolver2->convolve(nDatL, dpref_dex);
      ifpi.convolver2->convolve(nDatL, dpref_dp0);
      ifpi.convolver2->convolve(nDatL, dpref_dp1);
      ifpi.convolver2->convolve(nDatL, dpref_dp2);
      ifpi.convolver2->convolve(nDatL, dhtr_dech);
      ifpi.convolver2->convolve(nDatL, dhtr_derh);

      ifpi.convolver2->updatePSF(nDatL,dtr1_decl);
      ifpi.convolver2->convolve(nDatL, dltr_decl);
      
      ifpi.convolver2->updatePSF(nDatL,dtr1_derl);
      ifpi.convolver2->convolve(nDatL, dltr_derl);
      
      ifpi.convolver2->updatePSF(nDatL,dltr_dech);
      ifpi.convolver2->convolve(nDatL, dpref_dm1); // contains dltr_dech

      
      for(int ii=0; ii<nDatL; ++ii){
	dhtr_dech[ii] += dpref_dm1[ii]; // we had stored dltr_dech
	dpref_dm1[ii] = synl[ii] / m[9];
      }



      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal)); 
      for(int ii=0; ii<nDatH; ++ii)
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      
      for(int ii=0; ii<nDatL; ++ii)
	r[ii+nDatH] = (T(dl[ii]) - syn[nDatH+ii]) / (sigl[ii] * scl);
       

      
      // --- scale J --- //
      
      for(int ii = 0; ii<nPar; ++ii){
	if(!this->Pinfo[ii].fixed){
	  T const iScl = this->Pinfo[ii].scale / scl;
	  T* const __restrict__ iJ = J + ii*nDat;
	  
	  for(int ww = 0; ww<nDatH; ++ww)
	    iJ[ww] *=  iScl / sig[ww];
	  
	  for(int ww = 0; ww<nDatL; ++ww)
	    iJ[ww+nDatH] *=  iScl / sigl[ww]; 
	}
      }
      

      // --- get Chi2 --- //
      
      return this->getChi2(nDat, r);  
    }
    
    
    // ***************************************** //
    
    
  };
  
}

#endif

