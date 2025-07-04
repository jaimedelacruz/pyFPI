#ifndef LMHPP
#define LMHPP
/* ---
   
   Levenberg Marquardt algorithm

   Originally included in pyMilne but here
   modified in 2025 for the FPI calculations
   
   The container_base class can be inherited to implement
   each problem and pack all the required data.
   In this case, the implementation is done in
   lm_containers.hpp and lm_containers_laser.hpp

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2020, 2025)

   Modifications:
      - 2025-05: Implemented the container_base approach to make
                 the tool usable for different problems.
      - 2025-05: JdlCR: allow for fixed parameters.
      
   --- */

#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>

#include "math.hpp"
#include "fpi.hpp"

namespace lm{

  
  // ***************************************** //

  template<typename T>
  struct Par{
    bool isCyclic;
    bool limited;
    bool fixed;
    
    T scale;
    T limits[2];

    Par(): isCyclic(false), limited(false), fixed(false), scale(1.0), limits{0,0}{};
    Par(bool const cyclic, bool const ilimited, bool const ifixed, T const scal, T const mi, T const ma):
      isCyclic(cyclic), limited(ilimited), fixed(ifixed), scale(scal), limits{mi,ma}{};

    Par(Par<T> const& in): isCyclic(in.isCyclic) ,limited(in.limited), fixed(in.fixed), \
			   scale(in.scale), limits{in.limits[0], in.limits[1]}{};

    Par<T> &operator=(Par<T> const& in)
    {
      isCyclic = in.isCyclic, limited = in.limited, fixed = in.fixed, scale=in.scale, \
	limits[0]=in.limits[0], limits[1]=in.limits[1];
      return *this;
    }

    inline void Normalize(T &val)const{val /= scale;};
    
    inline void Scale(T &val)const{val *= scale;};
    
    inline void Check(T &val)const{
      if(!limited) return;
      if(isCyclic){
	if(val > limits[1]) val -= 3.1415926f;
	if(val < limits[0]) val += 3.1416026f;
      }
      val = std::max<T>(std::min<T>(val, limits[1]),limits[0]);
    }
    
    inline void CheckNormalized(T &val)const{
      if(!limited) return;
      Scale(val);
      Check(val);
      Normalize(val);
    }
    
  };
  
  // ***************************************** //
  
  template<typename T, typename U>
  struct container_base{
    int nDat;
    const T* const x;
    const U*  d;
    const T* const sig;
    const std::vector<Par<T>> &Pinfo;

    
    
    inline container_base(int const nd, const T* const xin, const U* const din,\
			  const T* const sigin, std::vector<Par<T>> const& Pi):
      nDat(nd), x(xin), d(din), sig(sigin), Pinfo(Pi){};


    virtual T fx(int const nPar, const T* __restrict__ m_in,
		 T* const __restrict__ syn, T* const __restrict__ r)const = 0;
    
    virtual T fx_dx(int const nPar, const T* __restrict__ m_in,
		    T* const __restrict__ syn, T* const __restrict__ r, T* const __restrict__ J)const = 0;


    static inline T getChi2(int const n, const T* const __restrict__ var)
    {
      T sum = T(0);
      for(int ii=0; ii<n; ++ii)
	sum += mth::SQ<T>(var[ii]);
      return sum;
    }

    virtual inline ~container_base(){};
    
  };
  

  // ***************************************** //

  template<typename T>
  T getChi2(int const nDat, const T* const __restrict__ r)
  {
    double sum = 0.0;

    for(int ii=0; ii<nDat;++ii)
      sum += T(mth::SQ(r[ii]));
    
    return static_cast<T>(sum);
  }


  // **************************************** // 
  
  template<typename T,typename U>
  struct LevMar{
    int nPar;
    std::vector<T> diag;
    std::vector<Par<T>> Pinfo;

    void set(int const& iPar){
      nPar = iPar;
      diag = std::vector<T>(iPar,0.0);
      Pinfo = std::vector<Par<T>>(iPar, Par<T>());
    }
    
    LevMar(): nPar(0), diag(), Pinfo(){};
    LevMar(int const &nPar_i):LevMar(){set(nPar_i);}

    LevMar(LevMar<T,U> const& in): LevMar(){nPar = in.nPar, diag=in.diag, Pinfo = in.Pinfo;}
    
    LevMar<T,U> &operator=(LevMar<T,U> const& in){nPar = in.nPar, diag=in.diag(), Pinfo = in.Pinfo; return *this;}

    static inline T checkLambda(T const val, T const mi, T const ma){return std::max<T>(std::min<T>(ma, val), mi);}


    // ------------------------------------------------------------------------------ //

    T getCorrection(container_base<T,U> const& myData, T* __restrict__ m, const T* __restrict__ Jin, \
		    T* __restrict__ syn, T* __restrict__ r, T const iLam)const
    {
      
      // --- Simplify the notation --- //
      
      using Mat = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
      using Vec = Eigen::Matrix<T, Eigen::Dynamic, 1>;
      

      // --- Define some quantities --- //

      int const nDat = myData.nDat;
      int const cPar = nPar;

      
      // --- Check which parameters are not fixed --- //

      int nfree = 0;
      for(int ii=0; ii<cPar; ++ii){
	if(!Pinfo[ii].fixed) nfree += 1;
      }


      // --- Copy non-fixed subparameter space --- //
      
      Mat J(nfree, nDat); J.setZero();
      
      int k = 0;
      for(int ii=0;ii<cPar;++ii){
	if(!Pinfo[ii].fixed)
	  std::memcpy(&J(k++,0),Jin+ii*nDat,nDat*sizeof(T));
      }
      
      
      // --- Allocate Linear system --- //
      
      Mat A(nfree, nfree); A.setZero();
      Vec B(nfree); B.setZero();
      
      
      
      // --- get Hessian matrix --- //
       
      for(int jj = 0; jj<nfree; ++jj){

	// --- Compute left-hand side of the system --- //

	const T* const __restrict__ Jj = &J(jj,0);

	for(int ii=0; ii<=jj; ++ii){
	  double sum = 0.0;
	  
	  const T* const __restrict__ Ji = &J(ii,0);
	  
	  for(int ww=0; ww<nDat; ++ww) sum += Jj[ww]*Ji[ww];
	  A(jj,ii) = A(ii,jj) = static_cast<T>(sum);
	}//ii
	
		
	// --- Compute right-hand side of the system --- //
	
	double sum = 0.0;

	for(int ww = 0; ww<nDat; ww++) sum += Jj[ww] * r[ww];
	B[jj] = static_cast<T>(sum); 
	
	
	A(jj,jj) *= 1.0+iLam;
      } // jj

      
      // --- Solve linear system to get solution --- //
      
      Eigen::ColPivHouseholderQR<Mat> sy(A); // Also rank revealing but much faster than SVD
      //Eigen::BDCSVD<Mat> sy(A,Eigen::ComputeThinU | Eigen::ComputeThinV);
      sy.setThreshold(1.e-14);
      
      Vec dm = sy.solve(B);

      // --- Copy parameters from non-fixed subspace to a nPar vector --- //

      Vec dx(cPar); dx.setZero();
      k = 0;
      
      for(int ii=0; ii<cPar; ++ii){
	if(!Pinfo[ii].fixed){
	  dx[ii] = dm[k++];
	}
      }
      
      
      // --- add to model and check parameters --- //

      for(int ii =0; ii<cPar; ++ii){
	m[ii] += dx[ii];
	Pinfo[ii].CheckNormalized(m[ii]);
      }   
      
      return myData.fx(nPar, m, syn, r);
    }
    
    // ------------------------------------------------------------------------------ //
    
    T getStep(container_base<T,U> const& myData, T* __restrict__ m, const T* __restrict__ J,
	      T* __restrict__ syn, T* __restrict__ r, T &iLam, bool braket, T const maxLam,
	      T const minLam)const{

      // if(!braket){
      return getCorrection(myData, m, J, syn, r, iLam);
	// }
      
      
    }
    
    // ------------------------------------------------------------------------------ //

    
    T fitData(container_base<T,U> &myData, int const nDat,  U* __restrict__ isyn, T* __restrict__ m, \
	      int const max_iter = 20, T iLam = sqrt(10.0), T const Chi2_thres = 1.0, \
	      T const fx_thres = 2.e-3, int const delay_braket = 2, bool verbose = true)const
    {
      static constexpr T const facLam = 3.1622776601683795;
      static constexpr T const maxLam = 1000;
      static constexpr T const minLam = 1.e-4;
      static constexpr int const max_n_reject = 4;
      

      // --- Check initial Lambda value --- //

      iLam = checkLambda(iLam, minLam, maxLam);

      
      // --- Init temp arrays and values--- //
      
      int const cPar = nPar;
      T bestChi2     = 1.e32;
      T     Chi2     = 1.e32;
      
      T* __restrict__ bestModel  = new T [cPar]();
      T* __restrict__ bestSyn    = new T [nDat]();      
      T* __restrict__ syn        = new T [nDat]();      

      T* __restrict__     J      = new T [cPar*nDat]();
      T* __restrict__     r      = new T [nDat]();


      
      // --- Work with normalized quantities --- //

      for(int ii =0; ii<cPar; ++ii){
	Pinfo[ii].Check(m[ii]);
	Pinfo[ii].Normalize(m[ii]);
      }
      
      std::memcpy(bestModel,  m, cPar*sizeof(T));

      
      // --- get derivatives and init Chi2 --- //

      bestChi2 = myData.fx_dx(nPar, bestModel, bestSyn, r, J);

      
      // --- Init iteration --- //

      if(verbose){
	fprintf(stderr, "\nLevDer::fitData: [Init] Chi2=%13.5f\n", bestChi2);
      }


      
      // --- Iterate --- //

      int iter = 0, n_rejected = 0;
      bool quit = false, tooSmall = false;
      T oLam = 0, dfx = 0;

      while(iter < max_iter){
	
	oLam = iLam;
	std::memcpy(m, bestModel, nPar*sizeof(T));

	
	// --- Get model correction --- //

	Chi2 = getStep(myData, m, J, syn, r, iLam, false, minLam, maxLam);


	// --- Did Chi2 improve? --- //

	if(Chi2 < bestChi2){

	  oLam = iLam;
	  dfx = (bestChi2 - Chi2) / bestChi2;
	  
	  bestChi2 = Chi2;
	  std::memcpy(bestModel,   m, cPar*sizeof(T));
	  std::memcpy(bestSyn  , syn, nDat*sizeof(T));

	  if(iLam > 1.0001*minLam)
	    iLam = checkLambda(iLam/facLam, minLam, maxLam);
	  else
	    iLam = 10*minLam;
	    
	  if(dfx < fx_thres){
	    if(tooSmall) quit = true;
	    else tooSmall = true;
	  }
	  
	  n_rejected = 0;
	}else{
	  
	  // --- Increase lambda and re-try --- //
	  
	  iLam = checkLambda(iLam*mth::SQ<T>(facLam), minLam, maxLam);
	  n_rejected += 1;
	  if(verbose)
	    fprintf(stderr,"LevMar::fitData: Chi2=%13.5f > %13.5f -> Increasing lambda %f -> %f\n", Chi2, bestChi2, oLam, iLam);
	  
	  if(n_rejected<max_n_reject) continue;

	}
	
	// --- Check what has happened with Chi2 --- //
	
	if(n_rejected >= max_n_reject){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: maximum number of rejected iterations reached, finishing inversion");
	  break;
	}

	if(verbose)
	  fprintf(stderr, "LevMar::fitData [%3d] Chi2=%13.5f, lambda=%e\n", iter, Chi2, oLam);

	if(bestChi2 < Chi2_thres){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: Chi2 (%f) < Chi2_threshold (%f), finishing inversion", bestChi2, Chi2_thres);
	  break;
	}

	if(quit){
	  if(verbose)
	    fprintf(stderr, "LevMar::fitData: Chi2 improvement too small for 2-iterations, finishing inversion\n");
	  break;
	}
	
	iter++;
	if(iter >= max_iter){
	  break;
	}

	// --- compute gradient of the new model for next iteration --- //

	std::memcpy(m, bestModel, cPar*sizeof(T));
	myData.fx_dx(nPar, m, syn, r, J);
      }
      
      std::memcpy(m, bestModel, cPar*sizeof(T));
      //std::memcpy(syn, bestSyn, nDat*sizeof(T));
      for(int ii=0; ii<nDat; ++ii){
	isyn[ii] = U(bestSyn[ii]);
      }
      
      
      for(int ii=0; ii<cPar; ++ii)
	Pinfo[ii].Scale(m[ii]);
      
      
      // --- Clean-up --- //

      delete [] bestModel;
      delete [] bestSyn;
      delete [] r;
      delete [] J;
      delete [] syn;

      return bestChi2;
    }
    
  };
  
}


#endif
