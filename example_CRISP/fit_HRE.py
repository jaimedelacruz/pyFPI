import numpy as np
import LevMar as lm
import satlas
import matplotlib.pyplot as plt; plt.ion()
import pyFPI

# ***************************************************************************

def convolve(var, tr):
    
    n = len(var)
    n1 = len(tr)
    npad = n + n1
    
    if((n1//2)*2 != n1):
        npad -= 1
        off = 1
    else:
        off = 0
    
    # Pad arrays using wrap around effect
    pvar = np.empty(npad, dtype='float64')
    pvar[0:n] = var
    pvar[n:n+n1//2] = var[-1]
    pvar[n+n1//2::] = var[0]
    
    ptr = np.zeros(npad, dtype = 'float64')
    ptr[0:n1] = tr / np.sum(tr)
    ptr = np.roll(ptr, -n1//2 + off)
    
    # FFT, convolve and FFT back
    return( (np.fft.irfft(np.fft.rfft(pvar) * np.conjugate(np.fft.rfft(ptr))))[0:n])


# ***************************************************************************

class Container:
    def __init__(self, wav, imean, ftsx, ftsy, fpi, no_prefilter=False):

        self.wav = np.copy(wav)
        self.imean = np.copy(imean)
        self.ftsx = np.copy(ftsx)
        self.ftsy = np.copy(ftsy)
        self.fpi = fpi
        self.npref = no_prefilter
        

# ***************************************************************************

def getPrefilter(wav, par, npref=False):
    if(not npref):
        tw = wav - par[1]
        return pyFPI.Prefilter(np.float64(wav), par[0], par[1], par[2], par[3], par[7], par[8], par[9]) #par[0] / (1.0 + np.abs(2.0*tw/par[2])**(2*par[3])) * (1.0 + (par[7] + tw*(par[8] + par[9]*tw))*tw)
    else:
        return par[0] * (1.0 + (par[7] + par[8]*wav + par[9]*wav**2)*wav)


# **************************************************************************

def getFX(par, myData, get_J = False):

    ftsx = myData.ftsx
    ftsy = myData.ftsy
    wav1 = myData.wav
    fpi = myData.fpi

    wav = wav1/par[5] + par[4]
    pftsy = getPrefilter(ftsx, par, npref = myData.npref)*np.interp(ftsx, ftsx, ftsy)

    dw = (ftsx[10]-ftsx[0]) * 0.1
    tw = (np.arange(ftsx.size, dtype='float64')-ftsx.size//2)*dw
    
    tr = fpi.dual_fpi_full(tw, erh=par[6])
    tr /= tr.sum()

    pftsy = convolve(pftsy,tr)

    return np.interp(wav,ftsx, pftsy)
    
    
# **************************************************************************

def fitPrefilterCRISP(wav, imean, init_fwhm=4.3, init_ncav=2.1, init_fts_shift=0.0, \
                      cw = None, sig = None, plot=False, no_prefilter=False, verbose=False):
    if(cw is None):
        cw = wav.mean()

    if(sig is None):
        sig = np.zeros(wav.size) + imean.mean() * 0.005
        
    # Init CRISP object
    fpi = pyFPI.CRISP(cw)
    
    # read atlas
    sa = satlas.satlas()
    ftsx, ftsy, ftsc = sa.getatlas(wav[0]-0.4, wav[-1]+0.4)

    # Init container for the Levenberg-Marquardt code
    myData = Container(wav-cw, imean, ftsx-cw, ftsy, fpi)

    
    # Init parameters
    
    p = np.zeros(10)
    p[0] = imean.max()
    p[1] = 0.0
    p[2] = init_fwhm
    p[3] = init_ncav
    p[4] = init_fts_shift
    p[5] = 1.0
    p[6] = -0.001
    p[7] = 0.00
    p[8] = 0.00
    p[9] = 0.00
    # Init parameter properties
    
    pinfo = [None]*10
    pinfo[0] = lm.Pinfo(scale=1.0, min = 0.0, max=5.0)
    pinfo[1] = lm.Pinfo(scale=1.0, min =-0.6, max=0.6)
    pinfo[2] = lm.Pinfo(scale=1.0, min = 1.0, max=9.6)
    pinfo[3] = lm.Pinfo(scale=1.0, min = 1.5, max=3.5)
    pinfo[4] = lm.Pinfo(scale=1.0, min = -0.6, max=0.6,is_fixed=False)
    pinfo[5] = lm.Pinfo(scale=1.0, min = 0.95, max=1.01,is_fixed=False)
    pinfo[6] = lm.Pinfo(scale=0.1, min = -0.03, max=0.03,is_fixed=False)
    pinfo[7] = lm.Pinfo(scale=0.1, min = -0.1, max=0.1,is_fixed=False)
    pinfo[8] = lm.Pinfo(scale=0.1, min = -0.1, max=0.1,is_fixed=False)
    pinfo[9] = lm.Pinfo(scale=0.1, min = -0.1, max=0.1,is_fixed=False)
    
    if(no_prefilter):
        pinfo[1].is_fixed = True
        pinfo[2].is_fixed = True
        pinfo[3].is_fixed = True
    else:
        pinfo[9].is_fixed = True
        pinfo[8].is_fixed = True

    chi2, pp, syn, J = lm.LevMar(getFX, p, imean, sig, pinfo, myData, Niter=20, \
                                 auto_derivatives=True, init_lambda=10.0, verbose=verbose, chi2_thres=1.0, \
                                 fx_thres=0.001, lmin=1.e-4, lmax=1.e4, lstep=10.0**0.5)
    
    if(plot):

        wav1 = (wav-cw)/pp[5] + cw
        wav1 += pp[4]

        idx = np.where((ftsx>wav1[0])&(ftsx<wav1[-1]))
        
        f, ax = plt.subplots(figsize=(4,2.7))
        ax.plot(wav1, imean, '.-',color='black', linewidth=0.5, mew=0.5, markersize=2, label=r'$<I_\mathrm{obs}>$')
        ax.plot(wav1, syn, color='orangered', linewidth=1.0,label='fit')
        ax.plot(ftsx[idx], getPrefilter(ftsx[idx]-cw, pp, npref=no_prefilter)/pp[0] * ftsy[idx], color='lightgray', linewidth=0.5, label=r'$FTS \cdot P(\lambda)$')
        
        ax.plot(wav1, getPrefilter(wav1-cw, pp, npref=no_prefilter)/pp[0],'--', color='gray', linewidth=1.0, label=r'$P(\lambda)$')
        ax.set_xlabel(r'wavelength [$\mathrm{\AA}$]')
        ax.set_ylabel(r'normalized intensity')
        ax.legend(frameon=False, loc='upper right', fontsize=6)
        ax.set_ylim(0.23, 1.1)
        f.subplots_adjust(bottom=0.18, left=0.13, right=0.96, top=0.92)
        #f.set_tight_layout(True)

        ax.set_title("average HRE spectrum fit", fontsize=9)
        f.savefig("fig_Imean.pdf", format='pdf', dpi=200)
        
    return pp, J, syn
    
# **************************************************************************
