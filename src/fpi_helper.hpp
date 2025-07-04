#ifndef FPIHELPHPP
#define FPIHELPHPP
/* ---
   Helper functions for the fpi class, including the initialization
   of the pupil angles for the full case.

   The routines fp_angles, hist2D and getpsi2 are adapted from
   Scharmer's ANA routines (Scharmer 2006).

   Coded by J. de la Cruz Rodriguez (ISP-SU,2025)
   --- */
#include <unsupported/Eigen/CXX11/Tensor>
#include <cmath>

namespace fpi{
  
  // ********************************************************** // 

  template<typename T>
  using Arr2D = Eigen::Tensor<T,2,Eigen::RowMajor>;

  template<typename T>
  using Arr1D = Eigen::Tensor<T,1,Eigen::RowMajor>;

  template<typename T>
  using Arr3D = Eigen::Tensor<T,3,Eigen::RowMajor>;
  
  // ********************************************************** //

  template<typename T>
  inline T max(long const N, const T* const __restrict__ var)
  {
    T mmax = var[0];
    for(long ii=0; ii<N; ++ii)
      mmax = std::max(var[ii],mmax);

    return mmax;
  }

  // ********************************************************** //

  template<typename T>
  inline T min(long const N, const T* const __restrict__ var)
  {
    T mmin = var[0];
    for(long ii=0; ii<N; ++ii)
      mmin = std::min(var[ii],mmin);

    return mmin;
  }
  
  // ********************************************************** // 

  template<typename T>
  inline T sum(Arr2D<T> const& var){

    const T* const __restrict__ data = &var(0,0);
    
    long const n = var.size();
    T sum = T(0);
    for(long ii=0; ii<n; ++ii) sum+= data[ii];
    return sum;
  }
    
  // ********************************************************** // 

  template<typename T>
  inline T sum(Arr3D<T> const& var){
    const T* const __restrict__ data = &var(0,0,0);

    long const n = var.size();
    T sum = T(0);
    for(long ii=0; ii<n; ++ii) sum+= data[ii];
    return sum;
  }
  // ********************************************************** //
  
  template<typename T>
  inline T max(Arr2D<T> const& var){
    const T* const __restrict__ data = &var(0,0);

    long const n = var.size();
    T mmax = data[0];
    for(long ii=1; ii<n; ++ii) mmax = std::max(data[ii],mmax);
    return mmax;
  }

  // ********************************************************** // 

  template<typename U, typename T>
  Arr2D<U> aperture(int const n,  T const rc)
  {
    Arr2D<U> res(n,n); res.setZero();
    U const n2 = n/2;
      
    for(int jj=0; jj<n;++jj){
      T const y = T(jj-n2);
      T const y2 = y*y;
      
      for(int ii=0; ii<n;++ii){
	T const x = T(ii-n2);
	T const r = std::sqrt(x*x + y2);
	
	res(jj,ii) = ((r<=rc)? U(1) : U(0));
      }
    }
    return res;
  }
  
  // ********************************************************** // 

  template<typename T>
  void meshgrid(int const ny, int const nx, Arr2D<T> &ymat, Arr2D<T> &xmat)
  {
    ymat = Arr2D<T>(ny,nx);
    xmat = Arr2D<T>(ny,nx);

    for(int yy=0; yy<ny; ++yy){
      for(int xx=0; xx<nx; ++xx){
	xmat(yy,xx) = T(xx);
	ymat(yy,xx) = T(yy);
      }
    }
  }
  
  // ********************************************************** //

  template<typename T>
  Arr1D<T> arange(long const N)
  {
    Arr1D<T> res(N);
    
    for(long ii=0; ii<N; ++ii)
      res[ii] = T(ii);
    
    return res;
  }
  
  // ********************************************************** // 

  template<typename T,typename U>
  Arr2D<T> fp_angles(int const n_ap, Arr2D<U> const& ap, T const FR,	\
		  T const tilt_angle)
  {
    constexpr const T PI =  3.1415926535897932384626433832;
    T const ap_sum = T(sum(ap));
    T const ir_mean =  T(1) /(T(2)* std::sqrt(ap_sum / PI));

    Arr2D<T> x; Arr2D<T> y;
    meshgrid(n_ap, n_ap, y, x);    

    T xap = T(0);
    T yap = T(0);

    for(int jj=0; jj<n_ap; ++jj){
      for(int ii=0; ii<n_ap; ++ii){
	xap += x(jj,ii)*ap(jj,ii);
	yap += y(jj,ii)*ap(jj,ii);
      }
    }
    
    xap /= ap_sum;
    yap /= ap_sum;

    x = (x - xap) * ir_mean;
    y = (y - yap) * ir_mean;
    
    T const alpha = tilt_angle / (T(2)*FR);

    y = y - FR*std::tan(alpha);
    y = x*x + y*y;
    
    int const nap2 = n_ap*n_ap;
    T* const __restrict__ yr = &y(0,0);

    for(int ii=0; ii<nap2;++ii)
      yr[ii] = std::atan(std::sqrt(yr[ii])/FR);
    
    return y;
  }
  
  // ********************************************************** // 

  template<typename T, typename U>
  void hist2D(Arr2D<U> const& ap, Arr2D<T> &betap_1, Arr2D<T> &betap_2, \
	      int const nrays1, int const nrays2, Arr2D<T> &n_betah,
	      Arr3D<T> &betah)
  {
    int const nx = ap.dimension(0);
    int const nap = std::round(sum(ap));
    Arr2D<T> dum(2,nap); dum.setZero();
    
    betap_1 = betap_1*betap_1;
    betap_2 = betap_2*betap_2;

    int k=0;
    
    for(int jj=0; jj<nx; ++jj){
      for(int ii=0; ii<nx; ++ii){
	if(ap(jj,ii) > U(0)){
	  T const ibetap1 = betap_1(jj,ii);
	  T const ibetap2 = betap_2(jj,ii);

	  dum(0,k) = ibetap1;
	  dum(1,k) = ibetap2;
	  
	  k+=1;
	}
      }
    }

    T const beta1_min = fpi::min(nap,&dum(0,0));
    T const beta2_min = fpi::min(nap,&dum(1,0));
    T const beta1_max = fpi::max(nap,&dum(0,0));
    T const beta2_max = fpi::max(nap,&dum(1,0));

    T const dbeta1 = (beta1_max-beta1_min)/T(nrays1);
    T const dbeta2 = (beta2_max-beta2_min)/T(nrays2);
    
    Arr1D<T> beta1 = beta1_min + dbeta1*T(0.5) + arange<T>(nrays1)*dbeta1;
    Arr1D<T> beta2 = beta2_min + dbeta2*T(0.5) + arange<T>(nrays2)*dbeta2;
    
    Arr1D<long> dum1(nap); dum1.setZero();
    Arr1D<long> dum2(nap); dum2.setZero();
    
    for(int ii=0;ii<nap; ++ii){
      dum1[ii] = long((dum(0,ii)-beta1_min)/dbeta1 - 1.e-10);
      dum2[ii] = long((dum(1,ii)-beta2_min)/dbeta2 - 1.e-10);
    }
    
    
    n_betah =  Arr2D<T>(nrays2,nrays1); n_betah.setZero();
    
    for(int ii=0; ii<nap;++ii){
      n_betah(dum2[ii], dum1[ii]) += T(1);
    }
    
    n_betah = n_betah / fpi::sum(n_betah);
    betah = Arr3D<T>(2,nrays2,nrays1);


    beta1 = beta1.sqrt();
    beta2 = beta2.sqrt();
    
    for(int n=0;n<nrays2;++n){
      for(int m=0; m<nrays1; ++m){
	betah(0,n,m) = beta1[m];
	betah(1,n,m) = beta2[n];
      }   
    }
    
    //betah = betah.sqrt();
  }
  
  // ********************************************************** // 

  template<typename T>
  Arr2D<T> get_psi2(int const nw, T const w0, const T* const dw, \
		    T const h,   Arr1D<T> beta1)
  {
    constexpr const T n = ft(1);
    constexpr const T two_PI_n =  T(2)*T(3.1415926535897932384626433832)*n;
    
    T const  c = two_PI_n * h;
    int const nrays = beta1.size();

    Arr2D<T> sin2p(nrays,nw);


    for(int nn=0; nn<nrays; ++nn){
      T const cbeta = c * std::cos(beta1[nn]);
      
      for(int ii=0; ii<nw; ++ii){
	sin2p(nn,ii) = sin(cbeta / (w0+dw[ii]));
      }
    }				

    sin2p = sin2p * sin2p;
    
    return sin2p;
  }

  // ********************************************************** // 
  
  template<typename T>
  Arr2D<T> get_psi2_der(int const nw, T const w0, const T* const dw,	\
			T const h,  Arr1D<T> beta1, Arr2D<T> &dsin2p)
  {
    constexpr const T n = ft(1);
    constexpr const T two_PI_n =  T(2)*T(3.1415926535897932384626433832)*n;
    
    T const  c = two_PI_n * h;
    constexpr T const dc = two_PI_n;
    
    int const nrays = beta1.size();
    
    Arr2D<T> sin2p(nrays,nw);
    dsin2p = fpi::Arr2D<T>(nrays,nw);
    

    for(int nn=0; nn<nrays; ++nn){
      
      T const cost =  std::cos(beta1[nn]);
      T const cbeta = c * cost;
      T const dcbeta = dc * cost;
      
      for(int ii=0; ii<nw; ++ii){
	T const iw = ft(1) / (w0+dw[ii]);
	T const arg = cbeta * iw;
	T const si = sin(arg);
	T const co = T(-2) * cos(arg);
	
	sin2p(nn,ii)  = si;
	dsin2p(nn,ii) = co * si * dcbeta * iw;
      }
    }				

    sin2p = sin2p * sin2p;
    
    return sin2p;
  }

  // ********************************************************** // 

}


#endif
