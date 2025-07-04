#ifndef LMCONTLASHPP
#define LMCONTLASHPP

/* ---

   Inherited subclasses implementing the model calculation for each case,
   in this case for laser data.

   Coded by J. de la Cruz Rodriguez (ISP-SU 2025)
   
   --- */

#include "math.hpp"
#include "fpi.hpp"
#include "lm.hpp"


namespace lm{

  // ***************************************** //

  template<typename T, typename U>
  struct container_lre_laser_fit: public container_base<T,U>{
    
    int Nreal;
    fpi::FPI const& ifpi;
    const T* const tw;

    ~container_lre_laser_fit(){};

    container_lre_laser_fit(int const nd, const T* const iwav, const U* const din, const T* const sigin, \
		      std::vector<Par<T>> const& Pi,fpi::FPI const& ifpi_in, const T* const itw):
      container_base<T,U>(nd,iwav,din,sigin,Pi), Nreal(nd), ifpi(ifpi_in), tw(itw){};

    // ----------------------------------------------------------------- //
    
    T fx(int const nPar, const T* __restrict__ m_in,
	 T* const __restrict__ syn, T* const __restrict__ r)const
    {
      
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;

      if(nDat != npsf){
	fprintf(stderr,"[error] lm::container_lre_laser_fit::fx: the psf grid must be identical to the data grid. (nDat == %d) != (npsf == %d)\n", nDat, npsf);
	exit(1);
      }
      

      // --- Copy model --- //
      
      T* const __restrict__ htr = new T [npsf]();
      T* const __restrict__ m = new T [nPar]();
      
      std::memcpy(m,m_in,nPar*sizeof(T));
      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles --- //
      
      ifpi.dual_fpi_conv_individual(npsf, tw, htr, syn, T(0), m[2], T(0), m[1], false, false);

      {
	T const im0 = m[0];
	for(int ii=0; ii<npsf; ++ii){
	  syn[ii] *= im0;
	}
      }

      
      
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDat; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }


      delete [] m;
      delete [] htr;
      
      return this->getChi2(nDat, r);
    }
    
    // ----------------------------------------------------------------- //
    
    T fx_dx(int const nPar, const T* const __restrict__ m_in,
	    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const
    {
       
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      
      if(nDat != npsf){
	fprintf(stderr,"[error] lm::container_lre_laser_fit::fx_dx: the psf grid must be identical to the data grid. (nDat == %d) != (npsf == %d)\n", nDat, npsf);
	exit(1);
      }
      // --- Copy model --- //
      
      T* const __restrict__ htr = new T [npsf]();
      T* const __restrict__ dtr = new T [4*npsf]();
      T* const __restrict__ m = new T [nPar]();
      
      std::memcpy(m,m_in,nPar*sizeof(T));
      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles and derivatives --- //
      
      ifpi.dual_fpi_conv_individual_der(npsf, tw, htr, syn, dtr, T(0), m[2], T(0), m[1], false, false);

      
      T const im0 = m[0];
      for(int ii=0; ii<npsf; ++ii){
	J[ii] = syn[ii];
	J[ii+1*nDat] = dtr[npsf*3+ii]*im0;
	J[ii+2*nDat] = dtr[npsf+ii]  *im0;
	
	syn[ii] *= im0;
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
      

      delete [] htr;
      delete [] dtr;
      delete [] m;
      
      return this->getChi2(nDat, r);
    }
  };

  
  // ***************************************** //

  
  template<typename T, typename U>
  struct container_hre_laser_fit: public container_base<T,U>{
    
    int Nreal;
    fpi::FPI const& ifpi;
    const T* const tw;

    ~container_hre_laser_fit(){};

    container_hre_laser_fit(int const nd, const T* const iwav, const U* const din, const T* const sigin, \
		      std::vector<Par<T>> const& Pi,fpi::FPI const& ifpi_in, const T* const itw):
      container_base<T,U>(nd,iwav,din,sigin,Pi), Nreal(nd), ifpi(ifpi_in), tw(itw){};

    // ----------------------------------------------------------------- //
    
    T fx(int const nPar, const T* __restrict__ m_in,
	 T* const __restrict__ syn, T* const __restrict__ r)const
    {
 
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;

      if(nDat != npsf){
	fprintf(stderr,"[error] lm::container_hre_laser_fit::fx: the psf grid must be identical to the data grid. (nDat == %d) != (npsf == %d)\n", nDat, npsf);
	exit(1);
      }
      

      // --- Copy model --- //
      
      T* const __restrict__ ltr = new T [npsf]();
      T* const __restrict__ m = new T [nPar]();
      
      std::memcpy(m,m_in,nPar*sizeof(T));
      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles --- //
      
      ifpi.dual_fpi_conv_individual(npsf, tw, syn, ltr, m[2], T(0), m[1], T(0), false, false);

      {
	T const im0 = m[0];
	for(int ii=0; ii<npsf; ++ii){
	  syn[ii] *= im0;
	}
      }
      
      
      
      // --- calculate residue --- //
      
      T const scl = sqrt(T(Nreal));
      
      for(int ii=0; ii<nDat; ++ii){
	r[ii] = (T(dat[ii]) - syn[ii]) / (sig[ii] * scl);
      }


      delete [] m;
      delete [] ltr;
      
      return this->getChi2(nDat, r);
    }
    
    // ----------------------------------------------------------------- //
    
    T fx_dx(int const nPar, const T* const __restrict__ m_in,
	    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const
    {
       
      int const npsf = ifpi.convolver->n1;
      int const nDat = this->nDat;
      

      if(nDat != npsf){
	fprintf(stderr,"[error] lm::container_hre_laser_fit::fx_dx: the psf grid must be identical to the data grid. (nDat == %d) != (npsf == %d)\n", nDat, npsf);
	exit(1);
      }
      
      // --- Copy model --- //
      
      T* const __restrict__ ltr = new T [npsf]();
      T* const __restrict__ dtr = new T [4*npsf]();
      T* const __restrict__ m = new T [nPar]();
      
      std::memcpy(m,m_in,nPar*sizeof(T));
      
      // --- Scale up model parameters --- //
      
      for(int ii=0; ii<nPar; ++ii){
	this->Pinfo[ii].Scale(m[ii]);
      }
      
      const U* const __restrict__ dat = this->d;
      const T* const __restrict__ sig = this->sig;
      
      
      
      // --- calculate transmission profiles and derivatives --- //
      
      ifpi.dual_fpi_conv_individual_der(npsf, tw, syn, ltr, dtr, m[2], T(0), m[1], T(0), false, false);

      
      T const im0 = m[0];
      
      for(int ii=0; ii<npsf; ++ii){
	J[ii] = syn[ii];
	J[ii+1*nDat] = dtr[npsf*2+ii]*im0;
	J[ii+2*nDat] = dtr[ii]  *im0;
	
	syn[ii] *= im0;
      }

      
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
      

      delete [] ltr;
      delete [] dtr;
      delete [] m;
      
      return this->getChi2(nDat, r);
    }
  };


}



#endif
