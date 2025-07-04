"""
Example / driver for a minimal dataset

Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot
import pyFPI as FF
#import numba
import matplotlib.pyplot as plt; plt.ion()
import fit_HRE as fh
import time
import satlas; sa = satlas.satlas()

# ******************************************************************************

#@numba.njit(fastmath=True)
def parab_fit(x, y):
    """
    Parabola fit to 3 points dataset
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """
    cf = np.empty(3, dtype='float64')

    d = x[0]
    e = x[1]
    f = x[2]
  
    yd = y[0]
    ye = y[1]
    yf = y[2]
  
    cf[1] = ((yf - yd) - (f**2 - d**2) * ((ye - yd) / (e**2 - d**2)))/ \
      ((f - d) - (f**2 - d**2) * ((e - d) / (e**2 - d**2)))
    cf[2] = ((ye - yd) - cf[1] * (e - d)) / (e**2 - d**2)
    cf[0] = yd - cf[1] * d - cf[2] * d**2
    
    return cf 

# ******************************************************************************

def initCavityMaps(cw, hwav, dh, lwav, dl, hc = 7871198.0, lc = 2955638.0):
    """
    Calculates an initial estimate of the HRE and LRE cavity maps
    cw: central wavelegth to convert from wav shift to cavity separation
    hwav: wavelength array of the HRE dataset (nwav, Angstroms)
    hdat: HRE data (ny,nx,nwav)
    lwav: wavelength array of the LRE dataset (nwav1, Angstroms)
    ldat: LRE data(ny,nx,nwav1)
      hc: cavity separation of the HRE (Angstroms)
      lc: cavity separation of the LRE (Angstroms)


    Comments: by commenting the @numba decorator, we can drop that dependency
    
    Coded by J. de la Cruz Rodriguez (ISP-SU, 2025)
    """

    ny, nx = dh.shape[0:2]
    hcmap = np.zeros((ny,nx), dtype=np.float32)
    lcmap = np.zeros((ny,nx), dtype=np.float32)
    nwav = hwav.size
    nwav1 = lwav.size

    himean = np.zeros(nwav)
    limean = np.zeros(nwav1)

    nsum = 0
    
    # get the HRE and LRE maps
    for jj in range(ny):
        for ii in range(nx):
            p = max(min(np.argmax(dl[jj,ii]), nwav1-1), 1)
            pp = parab_fit(lwav[p-1:p+2], dl[jj,ii,p-1:p+2])
            lcmap[jj,ii] = -0.5 * pp[1]/pp[2]
            
            p = max(min(np.argmin(dh[jj,ii]), nwav-1), 1)
            pp = parab_fit(hwav[p-1:p+2], dh[jj,ii,p-1:p+2])
            hcmap[jj,ii] = -0.5 * pp[1]/pp[2]

            himean += dh[jj,ii]
            limean += dl[jj,ii]
            
            nsum += 1
            
    # remove mean shift HRE
    himean /= nsum 
    p = np.argmin(himean)
    cc = parab_fit(hwav[p-1:p+2], himean[p-1:p+2])
    hcmap -=  -0.5 * cc[1]/cc[2]
    hcmap = - hcmap  * (hc / cw)
    
    # remove mean shift LRE
    limean /= nsum 
    p = np.argmax(limean)
    cc = parab_fit(lwav[p-1:p+2], limean[p-1:p+2])
    lcmap -=  -0.5 * cc[1]/cc[2]
    lcmap = - lcmap  * (hc / cw)

    
    # compensate the LRE cmap from the HRE shift
    lcmap -= hcmap * (lc / hc)

    return hcmap, lcmap
    
# ******************************************************************************

class Data:
    def _pad_data(self, pad_range):
        
        dwav = self.wav_raw[1]-self.wav_raw[0]
        npoints = int((pad_range)/dwav)
        if(npoints%2 == 0):
            npoints += 1

        ny,nx,nw = self.dat.shape
        dat = np.zeros((ny,nx,npoints), dtype=np.float32)
        wav1 = (np.arange(npoints, dtype=np.float64)-npoints//2)*dwav

        self.idx = np.zeros(nw,dtype=np.int32)

        
        for ii in range(nw):
            p = np.argmin(np.abs(wav1 - self.wav_raw[ii]))
            self.idx[ii] = p
            dat[:,:,p] = self.dat[:,:,ii]

        del self.dat
        del self.wav
        
        self.dat = dat
        self.wav = wav1 + self.wav_corr
        self.wav_raw = wav1
        
    # ---------------------------------------------------------------------
    
    def __init__(self, filename, pad_range=None, mask_tellurics = False, sig_level=5.e-3):

        # open fits file, it has 3 extensions: wav, compressed data, calibration_factors
        io = fits.open(filename, "readonly")

        # wavelength array
        self.wav_raw = io[0].data.astype(np.float64)

        # Data, must be transposed and converted to float32
        self.dat = np.ascontiguousarray(io[1].data.transpose((1,2,0)), dtype=np.float32)

        # Array of scaling factors: [wavelength corr, dmin, dmax, Iscl]
        tmp = io[2].data.astype(np.float64)
        self.wav_corr = tmp[0]*1.0
        self.dmin = tmp[1]*1.0
        self.dmax = tmp[2]*1.0
        self.dscl = tmp[3]*1.0 # 65535 / (dmax-dmin)

        # scale back data, it was scaled as (65535 / (dmax-dmin)) * (dat-dmin)
        self.dat *= 1.0/self.dscl
        self.dat += self.dmin
                        
        # Create calibrated wavelength array
        self.wav = self.wav_raw + self.wav_corr

        # indexes of the observed data (only relevant if we pad the data)
        self.idx = np.arange(self.wav.size)

        
        # Extend the data range by adding zeros on both sides (with zero weight!)
        if(pad_range is not None):
            self._pad_data(pad_range)

        # Mean spectrum
        self.imean = self.dat.mean(axis=(0,1))


        # define a sigma for the fits
        self.sig = np.zeros(self.wav.size) + 1.e60
        self.sig[self.idx] = np.power(self.imean[self.idx], 0.2)
        self.sig[self.idx] = sig_level * self.sig[self.idx] / self.sig[self.idx].mean()
        
        # mask tellurics
        if(mask_tellurics):
            self.sig[((self.wav+cw) > 6299.0927) & ((self.wav+cw)<6299.3054)] = 1.e60
            self.sig[((self.wav+cw) > 6301.8070) & ((self.wav+cw)<6302.1678)] = 1.e60
            self.sig[((self.wav+cw) > 6302.6420) & ((self.wav+cw)<6302.8384)] = 1.e60
            

        # Normalize the data so that the max intensity is around 1.0
        self.dat /= self.imean.max()
        self.imean /= self.imean.max()
            
    # ---------------------------------------------------------------------

    def getChi2(self, syn):
        chi2 = np.zeros(self.dat.shape[0:2])

        Nreal = 0
        for ww in range(self.wav.size):
            if(self.sig[ww] < 1.e10):
                chi2 += ((self.dat[:,:,ww] - syn[:,:,ww]) / (self.sig[ww]))**2
                Nreal += 1

        return chi2 / float(Nreal)
    
    # ---------------------------------------------------------------------

    def genPrefilterCurve(self, ph, cw, nthreads=8):
        pp = np.ascontiguousarray(ph.transpose((2,0,1)))
        pgain = np.ones(ph.shape[0:2])

        x, y, c = sa.getatlas(self.wav[0]-0.2+cw,self.wav[-1]+0.2+cw)
        fts = np.interp(self.wav+cw, x, y).astype(np.float32)
        
        return FF.PrefilterCube(self.wav, pgain,pp[3],pp[4],pp[5],pp[6],pp[7],pp[8], fts, nthreads=nthreads)
        
    # ---------------------------------------------------------------------

    
# ******************************************************************************
# MAIN PROGRAM: 
# ******************************************************************************

if __name__ == "__main__":

    # Some parameters
    nthreads = 10 # number of threads to use
    cw = 6302.0
    fpi_method = 2 # 0=perpendicular incidence, 1: conv approximation, 2: full calculation

    
    # Load HRE dataset, mask the 3 tellurics in the spectral range
    dh = Data("HRE_data.fits", mask_tellurics=True, sig_level=0.005)

    
    # Load LRE dataset, add zero padding all the way to +/- 8 Angstroms
    # to include the first LRE FSR peak on each side.
    dl = Data("LRE_data.fits", pad_range = 16.0, sig_level=0.0075)


    # get dimensions
    ny, nx = dh.dat.shape[0:2]
    
    
    # Init the cavity separation error maps using the 6301 line and the central peak of the LRE, in angstroms
    hcmap, lcmap = initCavityMaps(cw, dh.wav[131:181],dh.dat[:,:,131:181], dl.wav[455:507], dl.dat[:,:,455:507], hc = 7871198.0, lc = 2955638.0)
    

    # fit initial prefilter parameters from the HRE mean spectrum, and also the absolute wavelength calibration
    pp, J, fit = fh.fitPrefilterCRISP(dh.wav+cw, dh.imean, cw=cw, plot=True, no_prefilter=False)

    # add wavelength calibration to the wavelength arrays
    dh.wav += pp[4]; dh.wav_corr += pp[4]
    dl.wav += pp[4]; dl.wav_corr += pp[4]
    
    # init 2D fit parameters
    pinit_HRE = np.zeros((ny,nx,9))      
    pinit_HRE[:,:,0] = dh.imean.max()    # P_gain
    pinit_HRE[:,:,1] = hcmap             # cavity error map HRE
    pinit_HRE[:,:,2] = 0.0               # Delta Reflectivity HRE
    pinit_HRE[:,:,3] = pp[1]-pp[4]       # P_w0
    pinit_HRE[:,:,4] = pp[2]             # P_fwhm
    pinit_HRE[:,:,5] = pp[3]             # P_ncav
    pinit_HRE[:,:,6] = pp[7]             # P_0
    pinit_HRE[:,:,7] = 0.0               # P_1
    pinit_HRE[:,:,8] = 0.0               # P_2
    erl = np.zeros((ny,nx), dtype=np.float32)  # Assume that the error in LRE reflectivity is zero in the first HRE pass

    
    pinit_LRE = np.zeros((ny,nx,3))      
    pinit_LRE[:,:,0] = 20. * dl.imean.max()    # P_gain
    pinit_LRE[:,:,1] = lcmap                   # cavity error map LRE
    pinit_LRE[:,:,2] = 0.0                     # Delta Reflectivity LRE

    
    # what parameters are fixed in the HRE fit?
    fixed = np.zeros(9, dtype=np.int32)
    fixed[5] = 1 # P_ncav
    fixed[8] = 1 # P_2
    

    # Get the FTS atlas in the HRE range
    ftsx, ftsy, ftsc = sa.getatlas(dh.wav[0]-0.2+cw,dh.wav[-1]+0.2+cw)
    ftsx = ftsx.astype(np.float64) 
    ftsy = ftsy.astype(np.float64)


    
    # Perform the initial HRE fit
    t0 = time.time()
    hpar, hsyn = FF.fit_hre_CRISP(cw, pinit_HRE, dh.dat, dh.sig, dh.wav+cw, ftsx, ftsy, fixed, lcmap, \
                                erl , nthreads=nthreads, fpi_method=2)
    t1 = time.time()

    hchi2 = dh.getChi2(hsyn)
    print("[info] HRE fit ellapsed time -> {0:.1f}s, <Chi2> = {1:.2f}".format(time.time()-t0, hchi2.mean()))


    # Now the LRE fit, we need to provide an estimate of the HRE cavity map and reflectivity
    ech = np.ascontiguousarray(hpar[:,:,1]) + dl.wav_corr * (7871198.0 / cw )
    erh = np.ascontiguousarray(hpar[:,:,2])

    
    # Generate a prefilter*FTS atlas curve based on the prefilter parameters for the HRE.
    # This cube is generated in the LRE wavelength grid
    pref = dl.genPrefilterCurve(hpar, cw, nthreads=nthreads)
    

    # Fit the LRE
    t0 = time.time()
    lpar, lsyn = FF.fit_lre_CRISP(cw, pinit_LRE, dl.dat, dl.sig, dl.wav+cw, ech, erh, pref, fpi_method=fpi_method, nthreads=nthreads)
    t1 = time.time()

    lchi2 = dl.getChi2(lsyn)
    print("[info] LRE fit ellapsed time -> {0:.1f}s, <Chi2> = {1:.2f}".format(time.time()-t0, lchi2.mean()))



    # Now refine the HRE fits with the proper LRE parameters
    ecl = np.ascontiguousarray(lpar[:,:,1].astype(np.float32))
    erl = np.ascontiguousarray(lpar[:,:,2].astype(np.float32))
        
    t0 = time.time()
    hpar, hsyn = FF.fit_hre_CRISP(cw, hpar, dh.dat, dh.sig, dh.wav+cw, ftsx, ftsy, fixed, ecl, \
                                  erl, nthreads=nthreads, fpi_method=2)
    t1 = time.time()

    hchi2 = dh.getChi2(hsyn)
    print("[info] HRE fit ellapsed time -> {0:.1f}s, <Chi2> = {1:.2f}".format(time.time()-t0, hchi2.mean()))


    # Save results in fits file

    hdu = fits.HDUList([fits.PrimaryHDU(hpar), fits.ImageHDU(lpar)])
    hdu.writeto("HRE_LRE_inferred_parameters.fits", overwrite=True)
    print("[info] Derived parameters saved to -> HRE_LRE_inferred_parameters.fits")

    

    
    #
    # Make Plots
    #
    
    # convert to reflectity variations into total reflectivity in %
    fpi = FF.CRISP(cw) 
    href, lref = fpi.getReflectivities()
    hpar[:,:,2] = (hpar[:,:,2] + href) * 100.0
    lpar[:,:,2] = (lpar[:,:,2] + lref) * 100.0

    # center the prefilter
    pw0_mean =  hpar[:,:,3].mean()
    hpar[:,:,3] -= pw0_mean

    # remove the mean from the cavity maps and convert to nm
    hpar[:,:,1] = (hpar[:,:,1] - hpar[:,:,1].mean()) * 0.1
    lpar[:,:,1] = (lpar[:,:,1] - lpar[:,:,1].mean()) * 0.1
    

    
    f, ax = plt.subplots(nrows=2,ncols=4, figsize=(8,5), sharex=True, sharey=True, layout = "constrained")
    ax1 = ax.flatten()

    ims= [None]*8
    
    extent = np.float32((0,nx,0,ny))*0.044

    cmaps = ["RdGy","bone","bwr","bone","bone","bone","RdGy","bone"]
    vmax = [2., 93.4, 0.1, 5.34, -0.0243, 0.0058, 2.5, 84.0]
    vmin = [-2., 92.2, -0.1, 5.18, -0.033, -0.0031, -2.5, 82.6]
    labels = [ r'$\Delta C_{\mathrm{HRE}}$ [nm]', r'$R_{\mathrm{HRE}}$ [$\%$]', \
               r'$p_{w_0}-$'+'{0:.2f}'.format(pw0_mean+cw)+ r' [$\mathrm{\AA}$]', \
               r'$p_{fwhm}$ [$\mathrm{\AA}$]', r'$p_0$ [counts/$\mathrm{\AA}$]', \
               r'$p_1$ [counts/$\mathrm{\AA}^2$]', r'$\Delta C_{\mathrm{LRE}}$ [nm]', \
               r'$R_{\mathrm{LRE}}$ [$\%$]']

    k = 0
    for ii in range(1,8):
        if(fixed[ii] == 0):
            ims[k] = ax1[k].imshow(hpar[:,:,ii], interpolation='nearest', extent = extent, cmap=cmaps[k], vmax=vmax[k], vmin=vmin[k])
            k += 1
            
    for ii in range(1,3):
        ims[k] = ax1[k].imshow(lpar[:,:,ii], interpolation='nearest', extent = extent, cmap=cmaps[k], vmax=vmax[k], vmin=vmin[k])
        k+=1

    for ii in range(4):
        f.colorbar(ims[ii], orientation='horizontal', location='top', shrink=0.85, label=labels[ii])

    for ii in range(4,8):
        f.colorbar(ims[ii], orientation='horizontal', location='bottom', shrink=0.85, label=labels[ii])

    ax1[0].set_ylabel("y [arcsec]")
    ax1[4].set_ylabel("y [arcsec]")
    
    f.savefig("fig_results.pdf", dpi=250, format="pdf")
    
