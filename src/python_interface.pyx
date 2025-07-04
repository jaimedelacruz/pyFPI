"""
CYTHON interface for C++ fit_FPI tools.
Author: J. de la Cruz Rodriguez (ISP-SU, 2025)
"""
import cython
cimport numpy as np
from numpy cimport ndarray as ar
from numpy import zeros,  float64, float32, empty, power, arange, ones
from libcpp cimport bool
from libcpp.vector cimport vector
from cython.parallel import prange


# ********************************************************************************

__author__="J. de la Cruz Rodriguez"
__status__="Developing"

# ********************************************************************************

ctypedef double ft

# ********************************************************************************

cdef extern from "fftw3.h":
      void fftw_cleanup();

# ********************************************************************************

cdef extern from "math.hpp" namespace "mth":
    cdef void Interpolation "mth::interpolation_Linear<double>"(int N, const double* const x,  const double* const y,
			                                        int NN, const double* const xx, double* const yy);
     
# ********************************************************************************

cdef extern from "fpi.hpp" namespace "fpi":
    cdef cppclass FPI:
         ft hr, lr;
    
         FPI(ft icw, ft iFR, ft shr, ft slr, int NRAYS_HR, int NRAYS_LR)
         
         
         void dual_fpi_conv(int N1, const ft* const tw, ft* const tr,
		            ft erh, ft erl, ft ech,
		            ft ecl, bool normalize_tr)const;

         
         void dual_fpi_ray_der(int N1,  ft* const tw,
			       ft* const tr, ft* const dtr,
			       ft erh, ft erl,
			       ft ech, ft ecl,
                               ft angle, bool normalize_tr)const;
         
         void dual_fpi_ray(int N1,  ft* const tw,
		           ft* const tr, 
		           ft erh, ft erl,
		           ft ech, ft ecl,
                           ft angle, bool normalize_tr)const;
         
         void dual_fpi_conv_der(int N1,  ft* const tw,
			        ft* const tr, ft* const dtr,
			        ft erh, ft erl,
			        ft ech, ft ecl, bool normalize_tr)const;
         
         void dual_fpi_full(int N1,  ft* const tw,
			    ft* const tr,
                            ft erh, ft erl,
			    ft ech, ft ecl, bool normalize_tr)const;
         
         void dual_fpi_full_der(int N1,  ft* const tw,
			        ft* const tr, ft* const dtr,
                                ft erh, ft erl,
			        ft ech, ft ecl, bool normalize_tr)const;

         void dual_fpi_full_individual_der(int N1,  ft* const tw,
			                   ft* const htr, ft* const ltr,
                                           ft* const dtr, ft erh, ft erl,
			                   ft ech, ft ecl, bool normalize_ltr,
                                           bool normalize_htr)const;

         void dual_fpi_full_individual(int N1,  ft* const tw,
			               ft* const htr, ft* const ltr,
                                       ft erh, ft erl,
			               ft ech, ft ecl, bool normalize_ltr,
                                       bool normalize_htr)const;
         
         
         void dual_fpi_conv_individual_der(int N1,  ft* const tw,
			                   ft* const htr, ft* const ltr,
                                           ft* const dtr, ft erh, ft erl,
			                   ft ech, ft ecl, bool normalize_ltr,
                                           bool normalize_htr)const;

         
         void dual_fpi_conv_individual(int N1,  ft* const tw,
			               ft* const htr, ft* const ltr,
                                       ft erh, ft erl,
			               ft ech, ft ecl, bool normalize_ltr,
                                       bool normalize_htr)const;

        
         void dual_fpi_ray_individual_der(int N1,  ft* const tw,
			                   ft* const htr, ft* const ltr,
                                           ft* const dtr, ft erh, ft erl,
			                   ft ech, ft ecl, ft angle,
                                           bool normalize_ltr, bool normalize_htr)const;

         
         void dual_fpi_ray_individual(int N1,  ft* const tw,
			               ft* const htr, ft* const ltr,
                                       ft erh, ft erl,
			               ft ech, ft ecl, ft angle,
                                       bool normalize_ltr, bool normalize_htr)const;
         
         
         ft getFWHM()const;
         
         ft getFSR()const;

         void init_convolver(int ndata, int npsf);
         void init_convolver2(int ndata, int npsf);
         void set_reflectivities(ft ihr, ft ilr);
         ft get_HRE_reflectivity()const;
         ft get_LRE_reflectivity()const;

         
ctypedef FPI cFPI

# ********************************************************************************

cdef extern from "invert.hpp" namespace "fpi":
     cdef void invert_hre_crisp(long ny, long nx, long npar, long nwav, long nfts,
			        const ft* const fts_x, const ft* const fts_y, const float* const d, \
                                ft* const par, float* const syn, const ft* const wav, \
			        vector[FPI*] &fpis, const ft* const sig, const ft* const tw,
                                const int* const fixed, int fpi_method, bool no_pref,
                                const float* const ecl, const float* const erl) nogil;




     cdef void invert_lre_crisp(long ny, long nx, long npar, long nwav, \
			        const float* const d, ft* const par, float* const syn, const ft* const wav,\
			        vector[FPI*] &fpis, const ft* const sig, const ft* const tw, \
			        const float* const pref, const ft* const ech, const ft* const erh,
                                int fpi_method) nogil;
     

     
     cdef void invert_lre_crisp_laser(long ny, long nx, long npar, long nwav, \
			              const float* const d, ft* const par, float* const syn, const ft* const wav,\
			              vector[FPI*] &fpis, const ft* const sig, const ft* const tw);

     cdef void invert_hre_crisp_laser(long ny, long nx, long npar, long nwav, \
			              const float* const d, ft* const par, float* const syn, const ft* const wav,\
			              vector[FPI*] &fpis, const ft* const sig, const ft* const tw);


     cdef void invert_all_crisp(long ny, long nx, long npar, long nwavh, long nwavl, long nfts, \
			        const ft* const fts_x, const ft* const fts_y, const ft* const fts_yl, const float* const dh, const float* const dl,
			        ft* const par, float* const syn, const ft* const wavh, const ft* const wavl,	\
			        vector[FPI*] &fpis, const ft* const sigh, const ft* const sigl, const ft* const tw, const int* const fixed,
			        int fpi_method, ft dwgrid) nogil;
     
# ********************************************************************************

cdef extern from "prefilter.hpp" namespace "pref":
     cdef double prefilter "pref::prefilter<double>"(double wav, double pg, double cw, double fwhm, \
		                                     double ncav,double p0, double p1, double p2, double backgr) nogil;
     cdef double dprefilter "pref::Dprefilter<double>"(double wav, double pg, double cw, double fwhm, \
		                                     double ncav,double p0, double p1, double p2, double backgr, double* dp);
     
     cdef double apodization "pref::apodization<double>"(double tw, double cw, double fwhm);
     cdef double dapodization "pref::Dapodization<double>"(double tw, double cw, double fwhm, double &da_dcw, double &da_dfwhm);

# ********************************************************************************


@cython.boundscheck(False)
@cython.wraparound(False)
def PrefilterCube(ar[double,ndim=1] tw, ar[double,ndim=2] pg, ar[double,ndim=2] cw, ar[double,ndim=2] fwhm, ar[double,ndim=2] ncav, \
                  ar[double,ndim=2]  p0, ar[double,ndim=2] p1, ar[double,ndim=2] p2, ar[float,ndim=1] fts, int nthreads = 8):
    """
    Function Prefilter calculates an analytical prefilter curve according
    to the mathematical expression:
       Pref = pg / (1 + (2*dw/fwhm)**(2*ncav)) * (1 + p0*dw + p1*dw**2 + p2*dw**3)
       with dw = wav - cw

    Input:
      tw: 1D float64 array with the wavelength array (in Angstroms)
      pg: global scale factor (2D float64 array)
      cw: central wavelength of the prefilter (in Angstroms)  (2D float64 array)
    fwhm: FWHM of the prefilter (in Angstroms)  (2D float64 array)
    ncav: number of cavities of the prefilter (typically between 2 and 3)  (2D float64 array)
      p0: coefficient of the linear term of the polynomial component  (2D float64 array)
      p1: coefficient of the quadratic term of the polynomial component  (2D float64 array)
      p2: coefficient of the cubic term of the polynomial component  (2D float64 array)
nthreads: number of threads to use in the calculations (int, default = 8)
    Output:
      prefilter: 3D float32 array with the prefilter curve over the Fov

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """

    cdef long ny = cw.shape[0]
    cdef long nx = cw.shape[1]
    cdef long nwav = tw.size
    cdef long ipix = 0
    cdef long npix = nx*ny
    cdef long ww = 0
    
    cdef long xx = 0
    cdef long yy = 0

    cdef ar[float,ndim=3] pref = zeros((ny,nx,nwav), dtype='float32')
    cdef double* rpg = <double*>pg.data
    cdef double* rcw = <double*>cw.data
    cdef double* rfw = <double*>fwhm.data
    cdef double* rnc = <double*>ncav.data
    cdef double* rp0 = <double*>p0.data
    cdef double* rp1 = <double*>p1.data
    cdef double* rp2 = <double*>p2.data
    cdef double* rtw = <double*>tw.data
    cdef float* rpref = <float*>pref.data
    cdef float* rfts = <float*>fts.data
    
    with nogil:
        for ipix in prange(npix, num_threads=nthreads):
            if(rpg[ipix] > 1.e-2):
                for ww in range(nwav):    
                    rpref[ipix*nwav+ww] = float(prefilter(rtw[ww],rpg[ipix],rcw[ipix],rfw[ipix],rnc[ipix],rp0[ipix],rp1[ipix],rp2[ipix], 1.0))*fts[ww]
                    
    return pref

# ********************************************************************************

def Prefilter(ar[double,ndim=1] tw, double pg, double cw, double fwhm, double ncav, \
              double p0, double p1, double p2):
    """
    Function Prefilter calculates an analytical prefilter curve according
    to the mathematical expression:
       Pref = pg / (1 + (2*dw/fwhm)**(2*ncav)) * (1 + p0*dw + p1*dw**2 + p2*dw**3)
       with dw = wav - cw

    Input:
      tw: 1D float64 array with the wavelength array (in Angstroms)
      pg: global scale factor
      cw: central wavelength of the prefilter (in Angstroms)
    fwhm: FWHM of the prefilter (in Angstroms)
    ncav: number of cavities of the prefilter (typically between 2 and 3)
      p0: coefficient of the linear term of the polynomial component
      p1: coefficient of the quadratic term of the polynomial component
      p2: coefficient of the cubic term of the polynomial component
    
    Output:
      prefilter: 1D float64 array with the prefilter curve

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """
    cdef long nw = tw.size
    cdef long ii = 0

    cdef ar[double,ndim=1] res = zeros(nw,dtype='float64')

    for ii in range(nw):
        res[ii] = prefilter(tw[ii],pg,cw,fwhm,ncav,p0,p1,p2,1.0)

    return res        

# ********************************************************************************

def DPrefilter(ar[double,ndim=1] tw, double pg, double cw, double fwhm, double ncav, \
                     double p0, double p1, double p2):
    """
    Function DPrefilter calculates an analytical prefilter curve and its Jacobian according
    to the mathematical expression:
       Pref = pg / (1 + (2*dw/fwhm)**(2*ncav)) * (1 + p0*dw + p1*dw**2 + p2*dw**3)
       with dw = wav - cw

    Input:
      tw: 1D float64 array with the wavelength array (in Angstroms)
      pg: global scale factor
      cw: central wavelength of the prefilter (in Angstroms)
    fwhm: FWHM of the prefilter (in Angstroms)
    ncav: number of cavities of the prefilter (typically between 2 and 3)
      p0: coefficient of the linear term of the polynomial component
      p1: coefficient of the quadratic term of the polynomial component
      p2: coefficient of the cubic term of the polynomial component

    Output:
      prefilter: 1D float64 array with the prefilter curve
      J: 2D array (7, ntw) containing the derivatives relative to each
         of the parameters
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """
    cdef long nw = tw.size
    cdef long ii = 0

    cdef ar[double,ndim=1] res = zeros(nw,dtype='float64')
    cdef ar[double,ndim=2] Dres = zeros((nw,7),dtype='float64')

    for ii in range(nw):
        res[ii] = dprefilter(tw[ii],pg,cw,fwhm,ncav,p0,p1,p2,1.0,(<double*>Dres.data)+7*ii)

    return res,Dres

# ********************************************************************************

def Apodization(ar[double,ndim=1] tw, double cw, double fwhm):
    
    cdef int nw = tw.size
    cdef int ii = 0

    cdef ar[double,ndim=1] apod = ones(nw, dtype='float64')


    for ii in range(nw):
        apod[ii] = apodization(tw[ii], cw, fwhm)

    return apod

# ********************************************************************************

def DApodization(ar[double,ndim=1] tw, double cw, double fwhm):
    cdef int nw = tw.size
    cdef ar[double,ndim=1] apod = ones(nw, dtype='float64')
    cdef ar[double,ndim=2] Dapod = ones((2,nw), dtype='float64')

    cdef int ii = 0

    for ii in range(nw):
        apod[ii] = dapodization(tw[ii], cw, fwhm, Dapod[0,ii], Dapod[1,ii])
        
    return apod, Dapod

# ********************************************************************************

def fit_lre_CRISP(ft w0, ar[ft,ndim=3] par, ar[float,ndim=3] d, ar[ft,ndim=1] sig, \
                  ar[ft,ndim=1] wav, ar[ft,ndim=2] ech, ar[ft,ndim=2] erh, ar[float,ndim=3] pref, \
                  int fpi_method = 2, int nrays_hr = 5, int nrays_lr = 5, int nthreads = 8):

    
    cdef long ny = d.shape[0]
    cdef long nx = d.shape[1]
    cdef long nwav = wav.size
    cdef long npar = par.shape[2]
    cdef ft dw = wav[1] - wav[0]
    

    print("[info] fit_lre_CRISP: ny={0:d}, nx={1:d}, nwav={2:d}, dw_fts={3:f}, fpi_method={4:d}".format(ny,nx,nwav,dw,fpi_method))

    
    # --- Init fpi class ---
    
    cdef vector[cFPI*] fpis;
    cdef int ii = 0

    for ii in range(nthreads):
        fpis.push_back(new cFPI(w0, 165.0, 787.e4, 295.5e4, nrays_hr, nrays_lr));


    
    # --- Init the FFW3 convolver --- 
    
    for ii in range(nthreads):
        fpis[ii].init_convolver(nwav, nwav)


    # make prefilter

    cdef ar[ft,ndim=1] tw = wav - w0

    

    # --- perform data fits --- 

    cdef ar[float,ndim=3] syn = zeros((ny,nx,nwav),dtype=float32)



    # --- invert LRE data ---

    invert_lre_crisp(ny, nx, npar, nwav, <float*>d.data, <ft*>par.data, <float*>syn.data, <ft*>tw.data,
                     fpis, <ft*>sig.data, <ft*>tw.data, <float*>pref.data, <ft*>ech.data, <ft*>erh.data,
                     <int>fpi_method);



    # --- cleanup ---

    for ii in range(nthreads):
        del fpis[ii];

    fftw_cleanup()
        
    return par, syn
    
    
# ********************************************************************************


def fit_lre_CRISP_laser(ft w0, ar[ft,ndim=3] par, ar[float,ndim=3] d, ar[ft,ndim=1] sig, \
                        ar[ft,ndim=1] wav, int nrays_hr = 5, int nrays_lr = 5, int nthreads = 8):

    
    cdef long ny = d.shape[0]
    cdef long nx = d.shape[1]
    cdef long nwav = wav.size
    cdef long npar = par.shape[2]
    cdef ft dw = wav[1] - wav[0]
    

    print("[info] fit_lre_CRISP_laser: ny={0:d}, nx={1:d}, nwav={2:d}, dw_fts={3:f}".format(ny,nx,nwav,dw))

    
    # --- Init fpi class ---
    
    cdef vector[cFPI*] fpis;
    cdef int ii = 0

    for ii in range(nthreads):
        fpis.push_back(new cFPI(w0, 165.0, 787.e4, 295.5e4, nrays_hr, nrays_lr));


    
    # --- Init the FFW3 convolver --- 
    
    for ii in range(nthreads):
        fpis[ii].init_convolver(nwav, nwav)


    # --- Wavelength offset from reference
    
    cdef ar[ft,ndim=1] tw = wav - w0

    

    # --- perform data fits --- 

    cdef ar[float,ndim=3] syn = zeros((ny,nx,nwav),dtype=float32)



    # --- invert LRE data ---

    invert_lre_crisp_laser(ny, nx, npar, nwav, <float*>d.data, <ft*>par.data, <float*>syn.data, <ft*>tw.data,
                           fpis, <ft*>sig.data, <ft*>tw.data);



    # --- cleanup ---

    for ii in range(nthreads):
        del fpis[ii];
    fftw_cleanup()
    
    return par, syn

# ********************************************************************************

def fit_hre_CRISP_laser(ft w0, ar[ft,ndim=3] par, ar[float,ndim=3] d, ar[ft,ndim=1] sig, \
                        ar[ft,ndim=1] wav, int nrays_hr = 5, int nrays_lr = 5, int nthreads = 8):

    
    cdef long ny = d.shape[0]
    cdef long nx = d.shape[1]
    cdef long nwav = wav.size
    cdef long npar = par.shape[2]
    cdef ft dw = wav[1] - wav[0]
    

    print("[info] fit_hre_CRISP_laser: ny={0:d}, nx={1:d}, nwav={2:d}, dw_fts={3:f}".format(ny,nx,nwav,dw))

    
    # --- Init fpi class ---
    
    cdef vector[cFPI*] fpis;
    cdef int ii = 0

    for ii in range(nthreads):
        fpis.push_back(new cFPI(w0, 165.0, 787.e4, 295.5e4, nrays_hr , nrays_lr));


    
    # --- Init the FFW3 convolver --- 
    
    for ii in range(nthreads):
        fpis[ii].init_convolver(nwav, nwav)


    # --- Wavelength offset from reference
    
    cdef ar[ft,ndim=1] tw = wav - w0

    

    # --- perform data fits --- 

    cdef ar[float,ndim=3] syn = zeros((ny,nx,nwav),dtype=float32)



    # --- invert LRE data ---

    invert_hre_crisp_laser(ny, nx, npar, nwav, <float*>d.data, <ft*>par.data, <float*>syn.data, <ft*>tw.data,
                           fpis, <ft*>sig.data, <ft*>tw.data);



    # --- cleanup ---

    for ii in range(nthreads):
        del fpis[ii];
    fftw_cleanup()
    
    return par, syn

# ********************************************************************************
   
def fit_hre_CRISP(ft w0, ar[ft,ndim=3] par, ar[float,ndim=3] d, ar[ft,ndim=1] sig, \
                  ar[ft,ndim=1] wav, ar[ft,ndim=1] ftsx, ar[ft,ndim=1] ftsy, \
                  ar[int,ndim=1] fixed, ar[float,ndim=2] ecl, ar[float,ndim=2] erl, \
                  int fpi_method = 2, int nthreads=8, \
                  int nrays_hr = 5, int nrays_lr = 5, bool no_prefilter=False):

    cdef long ny = d.shape[0]
    cdef long nx = d.shape[1]
    cdef long nwav = wav.size
    cdef long npar = par.shape[2]
    cdef long nfts = ftsx.size
    cdef ft dw = (ftsx[10] - ftsx[0]) * 0.1

    cdef ar[ft,ndim=1] ftsx_cw = ftsx - w0
    cdef ar[ft,ndim=1] wav_cw = wav - w0

    print("[info] fit_hre_CRISP: ny={0:d}, nx={1:d}, nwav={2:d}, nfts={3:d}, dw_fts={4:f}, fpi_method={5:d}".format(ny,nx,nwav,nfts,dw, fpi_method))

    
    # --- Init fpi class, one per threat ---
    
    cdef vector[cFPI*] fpis;
    cdef int ii = 0

    
    for ii in range(nthreads):
        fpis.push_back(new cFPI(w0, 165.0, 787.e4, 295.5e4, nrays_hr, nrays_lr));

    
    
    # --- Estimate number of points for the fpi PSF and create tw array --- 

    cdef ft FSR = fpis[0].getFSR()
    cdef int npsf = round(2.25*fpis[0].getFSR() / dw)
    if(npsf > nfts):
        print("[warning] fit_hre_CRISP: the input FTS atlas does not cover +/- one FSR!")
        npsf = nfts
        
    
    if(npsf%2 == 0):
        npsf -= 1

    
    cdef ar[ft,ndim=1] tw = zeros(npsf)
    for ii in range(npsf):
        tw[ii] = <ft>(ii-npsf//2)*dw

    
    # --- Init the FFW3 convolver --- 
    
    for ii in range(nthreads):
        fpis[ii].init_convolver(nfts, npsf)


    # --- if no_prefilter, make sure you fix the prefilter parameters ---

    if(no_prefilter):
        fixed[3] = 1
        fixed[4] = 1
        fixed[5] = 1

        

    # --- perform data fits --- 

    cdef ar[float,ndim=3] syn = zeros((ny,nx,nwav),dtype=float32)


    invert_hre_crisp(ny,nx,npar,nwav,nfts,<ft*>ftsx_cw.data,<ft*>ftsy.data,<float*>d.data,<ft*>par.data,\
                     <float*>syn.data,<ft*>wav_cw.data,fpis,<ft*>sig.data, <ft*>tw.data, <int*>fixed.data,\
                     <int>fpi_method, <bool>no_prefilter, <float*>ecl.data, <float*>erl.data)
    
    
    # --- cleanup pointers ----

    for ii in range(nthreads):
        del fpis[ii]

    fftw_cleanup()
    
    return par, syn

# ********************************************************************************
   
def fit_all_CRISP(ft w0, ar[ft,ndim=3] par, ar[float,ndim=3] dh, ar[ft,ndim=1] sigh, \
                  ar[ft,ndim=1] wavh,  ar[float,ndim=3] dl, ar[ft,ndim=1] sigl, \
                  ar[ft,ndim=1] wavl, ar[ft,ndim=1] ftsx, ar[ft,ndim=1] ftsy, \
                  ar[ft,ndim=1] ftsyl, ar[int,ndim=1] fixed, double dwgrid = 0.0, \
                  int fpi_method = 2, \
                  int nthreads=8, int nrays_hr = 5, int nrays_lr = 5):

    cdef long ny = dh.shape[0]
    cdef long nx = dh.shape[1]
    cdef long nwavh = wavh.size
    cdef long nwavl = wavl.size
    cdef long npar = par.shape[2]
    cdef long nftsh = ftsx.size

    cdef ft dwh = (ftsx[10] - ftsx[0]) * 0.1

    cdef ar[ft,ndim=1] ftsx_cw = ftsx - w0
    cdef ar[ft,ndim=1] wav_cw = wavh - w0
    cdef ar[ft,ndim=1] wavl_cw = wavl - w0

    print("[info] fit_all_CRISP: ny={0:d}, nx={1:d}, fpi_method={2:d}".format(ny,nx,fpi_method))
    print("[info] fit_all_CRISP: HRE -> nwav={0:d}, dw={1:f}, nfts={2:d}".format(nwavh,dwh,nftsh))
    print("[info] fit_all_CRISP: LRE -> nwav={0:d}, dw={1:f}".format(nwavl,wavl[1]-wavl[0]))
    
    # --- Init fpi class, one per threat ---
    
    cdef vector[cFPI*] fpis;
    cdef int ii = 0

    
    for ii in range(nthreads):
        fpis.push_back(new cFPI(w0, 165.0, 787.e4, 295.5e4, nrays_hr, nrays_lr));

    
    
    # --- Estimate number of points for the fpi PSF and create tw array --- 

    cdef ft FSR = fpis[0].getFSR()
    cdef int npsf = round(2.25*fpis[0].getFSR() / dwh)
    if(npsf > nftsh):
        print("[warning] fit_hre_CRISP: the input FTS atlas does not cover +/- one FSR!")
        npsf = nftsh
        
    
    if(npsf%2 == 0):
        npsf -= 1

    
    cdef ar[ft,ndim=1] tw = zeros(npsf)
    for ii in range(npsf):
        tw[ii] = <ft>(ii-npsf//2)*dwh

    
    # --- Init the FFW3 convolver --- 
    
    for ii in range(nthreads):
        fpis[ii].init_convolver(nftsh, npsf)
        fpis[ii].init_convolver2(nwavl, nwavl)

        

    # --- perform data fits --- 

    cdef ar[float,ndim=3] syn = zeros((ny,nx,nwavh+nwavl),dtype=float32)

    invert_all_crisp(ny,nx,npar,nwavh,nwavl,nftsh,<ft*>ftsx_cw.data,<ft*>ftsy.data,<ft*>ftsyl.data,<float*>dh.data,<float*>dl.data, <ft*>par.data,\
                     <float*>syn.data,<ft*>wav_cw.data,<ft*>wavl_cw.data, fpis, <ft*>sigh.data, <ft*>sigl.data, <ft*>tw.data, <int*>fixed.data,\
                     <int>fpi_method, <double>dwgrid)
    
    
    # --- cleanup pointers ----

    for ii in range(nthreads):
        del fpis[ii]

    fftw_cleanup()
    
    return par, syn

# ********************************************************************************

cdef class CRISP:
    """
    Cython implementation of a CRISP-like dual-etalon system.
    The user can especify the cavity separation and F-ratio of the system
    to make calculations for a different system. The default parameters
    are taken from CRISP at the Swedish 1-m Solar Telescope.

    Note that the LRE cavity separation is automatically adjusted to
    follow the HRE. So the LRE profile is also moved by the ECH. So the ECL
    passed to the function is in practice ECL' = ECL + ECH*(lc/hc).
    This reference is very useful when deriving LRE cavity maps, as we
    automatically remove the imprint of the HRE cavity map (which is imprinted
    in the data). So the ECL is a "detune" from the HRE, wherever the HRE is.

    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    
    """
    cdef cFPI *cfpi

    # ------------------------------------------------------
    
    def __cinit__(self, double w0, double Fr=165.0, double hc = 787.e4, double lc = 295.5e4,
                  double hr = -1.0, double lr = -1.0, int nrays_hr = 5, int nrays_lr = 5, verbose = True):
        self.cfpi = new cFPI(w0, Fr, hc, lc, nrays_hr, nrays_lr);

        self.cfpi.set_reflectivities(hr,lr)
        
        if(verbose):
            print("[info] CRISP::__cinit__: C++ object initialized at lambda ={0:8.2f} nm, hr={3:f}, lr={4:f}, nrays_hr={1:d}, nrays_lr={2:d}".format(w0*0.1, nrays_hr, nrays_lr, self.cfpi.hr, self.cfpi.lr))

        
    # ------------------------------------------------------

    def __dealloc__(self):
        del self.cfpi
         
    # ------------------------------------------------------

    cpdef dual_fpi_full(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, \
                        ft ech = 0.0, ft ecl = 0.0, bool normalize_tr=False):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam and
        the tilt of the LRE (by 1/(2.0*Fr)). It performs a histogram of the angles over the pupil in order
        to speed up the calculations.

        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".

        """
        
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')

        self.cfpi.dual_fpi_full(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft>erh, \
                                <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_tr)
    
        return tr

    # ------------------------------------------------------

    cpdef dual_fpi_full_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0,\
                            ft ecl = 0.0, bool normalize_tr=False):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam and
        the tilt of the LRE (by 1/(2.0*Fr)). It performs a histogram of the angles over the pupil in order
        to speed up the calculations.
        
        Additionally, this function also calculates analytical derivatives of the transmission profile
        relative to the input reflectivities and cavity errors.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".
          dtr: a 4D array with the derivatives of tr relative to the 4 parameters (erh, erl, ech, ecl).
        
        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((4,nw),dtype='float64')

        self.cfpi.dual_fpi_full_der(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft*>dtr.data,\
                                    <ft>erh, <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_tr)
    
        return tr, dtr


    # ------------------------------------------------------

    cpdef dual_fpi_conv(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0,\
                        ft ecl = 0.0, bool normalize_tr=False ):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam.
        It neglects the tilt of the LRE and assumes axial-symmetry. Faster than dual_fpi_full.

        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".

        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')
    
        self.cfpi.dual_fpi_conv(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft>erh, \
                                <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_tr)
    
        return tr


    # ------------------------------------------------------

    cpdef dual_fpi_conv_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0,
                            ft ech = 0.0, ft ecl = 0.0, bool normalize_tr=False):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam.
        It neglects the tilt of the LRE and assumes axial-symmetry. Faster than dual_fpi_full.
        
        Additionally, this function also calculates analytical derivatives of the transmission profile
        relative to the input reflectivities and cavity errors.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".
          dtr: a 4D array with the derivatives of tr relative to the 4 parameters (erh, erl, ech, ecl).

        """
        
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((4,nw),dtype='float64')

        self.cfpi.dual_fpi_conv_der(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft*>dtr.data, <ft>erh, \
                                    <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_tr)
    
        return tr, dtr


    # ------------------------------------------------------

    cpdef dual_fpi(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0, \
                   ft ecl = 0.0, angle=0.0, bool normalize_tr=False):
        """
        Calculates the dual-etalon combined profile for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".

        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')
        
        self.cfpi.dual_fpi_ray(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft>erh, \
                               <ft>erl, <ft>ech, <ft>ecl, angle, <bool>normalize_tr )
    
        return tr
    

    # ------------------------------------------------------

    cpdef dual_fpi_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0, \
                       ft ecl = 0.0, angle=0.0, bool normalize_tr=False):
        """
        Calculates the dual-etalon combined profile for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           tr: the effective transmission profile of the system, evaluated at "tw".
          dtr: a 4D array with the derivatives of tr relative to the 4 parameters (erh, erl, ech, ecl).

        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] tr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((4,nw),dtype='float64')
                
        self.cfpi.dual_fpi_ray_der(<int>nw, <ft*>tw.data, <ft*>tr.data, <ft*>dtr.data, \
                                   <ft>erh, <ft>erl, <ft>ech, <ft>ecl, angle, <bool>normalize_tr)

        return tr, dtr

    # ------------------------------------------------------

    cpdef dual_fpi_individual(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0, \
                              ft ecl = 0.0, angle=0.0, bool normalize_ltr=False, bool normalize_htr=False):
        """
        Calculates the dual-etalon individual profiles for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".

        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
        
        self.cfpi.dual_fpi_ray_individual(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data, <ft>erh, \
                                          <ft>erl, <ft>ech, <ft>ecl, angle, <bool>normalize_ltr, \
                                          <bool>normalize_htr)
    
        return htr, ltr
    

    # ------------------------------------------------------

    cpdef dual_fpi_individual_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0, \
                                  ft ecl = 0.0, angle=0.0, bool normalize_ltr=False, bool normalize_htr=False):
        """
        Calculates the dual-etalon individual profiles for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).

        Additionally, this function returns the analytical derivatives of the transmission profiles
        relative to the input cavity errors and reflectivities. Note that the LRE also has a dependence on
        the ECH because we take the cavity separation of the HRE as the wavelength reference.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".
           dtr: derivative array (dhtr_derh, dltr_derl, dhtr_dech, dltr_decl, dltr_dech)
        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((5,nw),dtype='float64')
                
        self.cfpi.dual_fpi_ray_individual_der(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data, <ft*>dtr.data, \
                                              <ft>erh, <ft>erl, <ft>ech, <ft>ecl, angle, <bool>normalize_ltr,
                                              <bool>normalize_htr)

        return htr, ltr, dtr

    # ------------------------------------------------------

    cpdef dual_fpi_conv_individual(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0,\
                                   ft ecl = 0.0, bool normalize_ltr=False, bool normalize_htr=False ):
        """
        Calculates the dual-etalon individual profiles for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".
        """    
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
    
        self.cfpi.dual_fpi_conv_individual(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data, <ft>erh, \
                                           <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_ltr,
                                           <bool>normalize_htr)
    
        return htr, ltr


    # ------------------------------------------------------

    cpdef dual_fpi_conv_individual_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0,
                            ft ech = 0.0, ft ecl = 0.0, bool normalize_ltr=False, bool normalize_htr=False):
        """
        Calculates the dual-etalon individual profiles for a ray with incidence angle "angle" relative
        to the normal of the surface of the etalon (defaul = 0, perpendicular incidence).

        Additionally, this function returns the analytical derivatives of the transmission profiles
        relative to the input cavity errors and reflectivities. Note that the LRE also has a dependence on
        the ECH because we take the cavity separation of the HRE as the wavelength reference.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".
           dtr: derivative array (dhtr_derh, dltr_derl, dhtr_dech, dltr_decl, dltr_dech)
        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((5,nw),dtype='float64')

        self.cfpi.dual_fpi_conv_individual_der(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data,\
                                               <ft*>dtr.data, <ft>erh, <ft>erl, <ft>ech, <ft>ecl, \
                                               <bool>normalize_ltr, <bool>normalize_htr)
    
        return htr,ltr, dtr

    # ------------------------------------------------------

    cpdef dual_fpi_full_individual(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0, ft ech = 0.0,\
                                   ft ecl = 0.0, bool normalize_ltr=False, bool normalize_htr=False ):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam and
        the tilt of the LRE (by 1/(2.0*Fr)). It performs a histogram of the angles over the pupil in order
        to speed up the calculations.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".
        """ 
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
    
        self.cfpi.dual_fpi_full_individual(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data, <ft>erh, \
                                           <ft>erl, <ft>ech, <ft>ecl, <bool>normalize_ltr,
                                           <bool>normalize_htr)
    
        return htr, ltr


    # ------------------------------------------------------

    cpdef dual_fpi_full_individual_der(self, ar[ft,ndim=1] tw, ft erh=0.0, ft erl = 0.0,
                            ft ech = 0.0, ft ecl = 0.0, bool normalize_ltr=False, bool normalize_htr=False):
        """
        Calculates the dual-etalon combined profile including the convergence of the telecentric beam and
        the tilt of the LRE (by 1/(2.0*Fr)). It performs a histogram of the angles over the pupil in order
        to speed up the calculations.

        Additionally, this function returns the analytical derivatives of the transmission profiles
        relative to the input cavity errors and reflectivities. Note that the LRE also has a dependence on
        the ECH because we take the cavity separation of the HRE as the wavelength reference.
        
        Input:
            tw: wavelength offset grid [Angstroms] used to compute the profile. 1D float64 array.
           erh: HRE reflectivity error (fraction). This float number is added to the nominal HRE reflectivity.
           erh: LRE reflectivity error (fraction). This float number is added to the nominal LRE reflectivity.
           ech: HRE cavity error [Angstrom]. This float number is added to the nominal HRE cavity separation.
           ecl: LRE cavity error [Angstrom]. This float number is added to the nominal LRE cavity separation.
                Note that the ecl is relative to the position of the HRE.
         angle: angle of incidence relative to the normal of the surface of the etalon (radians). 
        normalize_tr: if set, the returned profile is area-normalized (for convolutions). Bool-type.

        Output:
           htr: the transmission profile of the HRE, evaluated at "tw".
           ltr: the transmission profile of the HRE, evaluated at "tw".
           dtr: derivative array (dhtr_derh, dltr_derl, dhtr_dech, dltr_decl, dltr_dech)
        """
        cdef int nw = tw.size
        cdef ar[ft,ndim=1] htr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=1] ltr = zeros(nw,dtype='float64')
        cdef ar[ft,ndim=2] dtr = zeros((5,nw),dtype='float64')

        self.cfpi.dual_fpi_full_individual_der(<int>nw, <ft*>tw.data, <ft*>htr.data, <ft*>ltr.data,\
                                               <ft*>dtr.data, <ft>erh, <ft>erl, <ft>ech, <ft>ecl, \
                                               <bool>normalize_ltr, <bool>normalize_htr)
    
        return htr,ltr, dtr

    # ------------------------------------------------------

    cpdef getFSR(self):
         return self.cfpi.getFSR()
     
    # ------------------------------------------------------

    cpdef getFWHM(self):
         return self.cfpi.getFWHM()

    # ------------------------------------------------------

    def getReflectivities(self):
        """
        returns the HRE and LRE reflectivities
        """
        cdef double hr = self.cfpi.get_HRE_reflectivity()
        cdef double lr = self.cfpi.get_LRE_reflectivity()
        
        return  hr, lr
    
    # ------------------------------------------------------

# ********************************************************************************

