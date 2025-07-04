"""
Levenberg Marquart fitting class and helper tools

Coded by J. de la Cruz Rodriguez (ISP-SU 2021)

References:
This implementation follows the notation presented in:
de la Cruz Rodriguez, Leenaarts, Danilovic & Uitenbroek (2019): 
https://ui.adsabs.harvard.edu/abs/2019A%26A...623A..74D/abstract
but without l-2 regularization for the time being.

Original Levenberg-Marquardt algorithm references:
Levenberg (1944)
Marquardt (1963)


Dependences: NumPy

Modification history:
    2025-05-14: Fixed scaling of authoderivatives and
                allowed to fix parameter values in Pinfo.
                Also implemented external svd_thres value.
    

"""

import numpy as np

# *******************************************************

class Pinfo:
    """
    Helper class to store parameter scaling norms and limits
    Methods: checkLimits, scale, normalize
    """
    def __init__(self, scale = 1.0, min = None, max = None, is_cyclic = 0, is_fixed = 0):
        self.scl = scale
        self.min = min
        self.max = max
        self.is_cyclic  = is_cyclic
        self.is_fixed = is_fixed
        
    # --------------------------------------------------

    def checkLimits(self, value):
        """
        This function gets a parameter value and
        checks whether it is within the specified
        limits. If it isn't, the value will saturate
        at the limit value.
        """
        if(self.min is not None):
            value = np.maximum(value, self.min)
            
        if(self.max is not None):
            value = np.minimum(value, self.max)

        return value
    
    # --------------------------------------------------

    def scale(self, value):
        """
        This function scales a paramter value that 
        has been normalized with the scaling norm
        """
        return value*self.scl
    
    # --------------------------------------------------

    def normalize(self, value):
        """
        This function normalizes a parameter value 
        with a scaling norm
        """
        return value / self.scl
    
    
# *******************************************************

def ScalePars(pars, Pinfo):
    """
    Helper function that scales an array of parameters inplace
    pars: 1D numpy array of length Npar
    Pinfo: tuple/list of Pinfo objects of length Npar
    """
    
    pLen = pars.size
    piLen = len(Pinfo)

    if(piLen != pLen):
        print("[error] ScalePars: Error, parameter array has different length than Parameter info array: {0} != {1}".format(pLen, piLen))
        return 0

    for ii in range(pLen):
        pars[ii] = Pinfo[ii].scale(pars[ii])

    return 1

# *******************************************************

def NormalizePars(pars, Pinfo):
    """
    Helper function that normalizes an array of parameters inplace
    pars: 1D numpy array of length Npar
    Pinfo: tuple/list of Pinfo objects of length Npar
    """
    pLen = pars.size
    piLen = len(Pinfo)

    if(piLen != pLen):
        print("[error] NormalizePars: Error, parameter array has different length than Parameter info array: {0} != {1}".format(pLen, piLen))
        return 0

    for ii in range(pLen):
        pars[ii] = Pinfo[ii].normalize(pars[ii])

    return 1

# *******************************************************

def CheckPars(pars, Pinfo):
    """
    Helper function that checks an array of parameters inplace
    pars: 1D numpy array of length Npar
    Pinfo: tuple/list of Pinfo objects of length Npar
    """
    pLen = pars.size
    piLen = len(Pinfo)

    if(piLen != pLen):
        print("[error] CheckPars: Error, parameter array has different length than Parameter info array: {0} != {1}".format(pLen, piLen))
        return 0

    for ii in range(pLen):
        pars[ii] = Pinfo[ii].checkLimits(pars[ii])

    return 1

# *******************************************************

def _eval_fx(fx, x, pinfo, udat, auto_derivatives = False, get_J = False):
    """
    Internal helper function that evaluates the user provided function fx,
    and computes the Jacobian if needed. If the analytical form
    of the Jacobian is unknown, this routine can do if by finite 
    diferences
    """
    
    nPar = x.size
    xtmp = np.copy(x)
    
    status = ScalePars(xtmp, pinfo)

    
    if(get_J):

        #
        # The user does not provide a drivatives engine, compute them
        # automatically. Keep your fingers crossed
        #
        if(auto_derivatives):
            dpar = 1.E-4
            syn = fx(xtmp, udat)
            
            nObs = syn.size
            J   = np.zeros((nPar, nObs), dtype='float64')

            for ii in range(nPar):
                
                if(not pinfo[ii].is_fixed):
                    xtmp = np.copy(x)
                    xtmp[ii] += dpar
                    status = ScalePars(xtmp, pinfo)
                    left = fx(xtmp, udat)
                    
                    xtmp = np.copy(x)
                    xtmp[ii] -= dpar
                    status = ScalePars(xtmp, pinfo)
                    right = fx(xtmp, udat)
                    
                    J[ii] = (left - right) / (2*dpar*pinfo[ii].scl)
                
        else: # The user provides derivatives
            syn, J = fx(xtmp, udat, get_J=get_J)

        return syn, J

    else:
        #
        # No derivatives are requested
        #

        return fx(xtmp, udat)
        
# *******************************************************

def _getResidue(syn, o, s, pinfo, J = None):
    """
    Internal helper function that computes the
    residue and scales the Jacobian with sigma and 
    the sqrt of scaling factors
    """
    
    nPar = len(pinfo)
    nDat = o.size
    
    scl = np.sqrt(1.0/nDat)
    res = scl * (o-syn) / s
    
    
    if(J is not None):
        scl = scl/s
        for pp in range(nPar):
            J[pp] *= scl * pinfo[pp].scl
            
    return res
            
# *******************************************************

def _getChi2(res):
    """
    Helper function that computes Chi2 from the residue array
    """
    return (res*res).sum()

# *******************************************************

def _checkLambda(lamb, lmin, lmax, lstep):
    """
    Helper function that check the lambda parameter limits
    """
    
    if(lamb > lmax):
        return lmax
    elif(lamb < lmin):
        return lamb*lstep*lstep
    
    else:
        return lamb
    
# *******************************************************

def _solveLinearSVD(A,b, svd_thres = 1.e-14):
    """
    Resolution of a linear system of equation using SVD
    TODO: Singular value filtering for small values below
    svd_thres
    """
    U,s,Vh = np.linalg.svd(A)
    sthr = np.max(np.abs(s))*svd_thres

    # thresholding of singular values
    
    for ii in range(s.size):
        if(np.abs(s[ii]) < sthr):
            s[ii] = 0.0
        else:
            s[ii] = 1.0 / s[ii]
        
    c = np.dot(U.T,np.transpose(b))
    w = np.dot(np.diag(s),c)
    x = np.dot(Vh.conj().T,w)
    
    return x

# *******************************************************

def _computeNew(Jin, res, x, lamb, pinfo, svd_thres=1.e-14):
    """
    Helper function that computes the correction
    to the current estimate of the model for a given
    Jacobian matrix, residues array and lambda parameter
    """
    # Allocate linear system terms
    # A = J.T * J, where A is a symmetric matrix
    # b = J.T * res, where b is a vector

    nPar, nDat = Jin.shape

    
    # extract the Jacobian from non-fixed parameters
    
    nfree = 0
    for ii in range(nPar):
        if(pinfo[ii].is_fixed == 0):
            nfree += 1
            
    J = np.zeros((nfree,nDat))

    k = 0
    for ii in range(nPar):
        if(pinfo[ii].is_fixed == 0):
            J[k] = Jin[ii]
            k += 1


    # Create arrays to store the Hessian and RHS
    
    A = np.zeros((nfree,nfree), dtype='float64')
    b = np.zeros((nfree), dtype='float64')


    
    
    for jj in range(nfree):
        
        # Evaluate b = J.T * res
        b[jj] = (J[jj] * res).sum()
        
        for ii in range(jj,nfree): # Remember, it is sym!

            # Evaluate A = J.T * J
            tmp =  (J[jj]*J[ii]).sum()
                
            A[jj,ii] = tmp
            A[ii,jj] = tmp 

        # Apply diagonal damping to A matrix
        
        A[jj,jj] *= (1.0 + lamb)

        
    # Solve linear system for correction
    
    dx_sub = _solveLinearSVD(A, b, svd_thres=svd_thres)

    
    # insert correction for non-fixed parameters into a nPar-size array
    
    k=0
    dx = np.zeros(nPar)
    
    for ii in range(nPar):
        if(pinfo[ii].is_fixed == 0):
            dx[ii] = dx_sub[k]
            k+=1

    
    # Add correction to the current estimate of the model
    xnew = x + dx

    # Check parameter limits
    status = ScalePars(xnew, pinfo)
    status = CheckPars(xnew, pinfo)
    status = NormalizePars(xnew, pinfo)
    
    return xnew
    
    
# *******************************************************

def _getNewEstimate(fx, x, o, s, res, pinfo, udat, lamb, J, lmin, lmax, \
                    lstep, auto_derivatives = False, svd_thres=1.e-14):    
    """
    Wrapper helper function that computes a new estimate of the model
    and evaluates Chi2 for this estimate for a given J, res and lambda 
    parameter
    """
    
    
    # get nde estimate of the model for lambda parameter
    xnew = _computeNew(J, res, np.copy(x), lamb, pinfo, svd_thres=svd_thres)
    
    # get model prediction
    synnew = _eval_fx(fx, xnew, pinfo, udat, auto_derivatives=auto_derivatives, get_J = False)
    
    # residue, no J scaling this time as it is already done
    res_new = _getResidue(synnew, o, s, pinfo)
    new_chi2 =  _getChi2(res_new)
    
    return new_chi2, xnew, lamb

# *******************************************************

def LevMar(fx, par_in, obs_in, sig_in, pinfo, udat, Niter = 20, init_lambda=10.0, \
           lmin = 1.e-4, lmax=1.e4, lstep = 10**0.5, chi2_thres=1.0, \
           fx_thres=0.001, auto_derivatives = False, verbose = True, n_reject_max = 4,\
           svd_thres = 1.e-14):
    """
    Levenberg-Marquard based fitting routine for non-linear models
    Coded by J. de la Cruz Rodriguez (ISP-SU 2021)
    
    Input:
          fx: a user provided function that taxes as input fx(pars, user_data, get_J = True/False)
      par_in: a numpy array with the initial estimate of the model parameters (length=Npar). It will be flattened.
      obs_in: a numpy array with the data to be fitted of length (Ndat). It will be flattened internally.
      sig_in: a numpy array with the noise estimate for each data point (length Ndat).
       pinfo: a list of Pinfo objects of length (Npar), containing the scaling norm and the parameter limits (if any).
        udat: User provided data. This variable can be anyhing (a struct?) with data that will be passed to fx as an argument. The user can pack here as much info as needed.
    Optional:
           Niter: Maximum number of iterations (typically Niter=20)
     init_lambda: Initial value of the lambda parameter that scales the diagonal of the Hessian. Typically > 1.0 when starting the fitting process.
            lmin: Minimum value of lambda allowed (default 1.e-4)
            lmax: Maximum value of lambda allowed (default 1.e+4)
           lstep: step in lambda between iterations (lambda is divided by this number, default sqrt(10))
      chi2_thres: stop the iterations if Chi2 goes below this value. If the noise estimate is correct, the threshold should be around 1.
        fx_thres: stop iterating if the relative change in Chi2 is below this threshold for two consecutive iterations
auto_derivatives: Compute the derivatives automatically using centered finite differences (True/False default)
         verbose: Print iteration information
    n_reject_max: maximum number of consecutive rejected iterations before stopping the iterations. The lambda parameter will be increased after each rejection.
       svd_thres: sets the cutoff threshold for setting singular values to zero.
     
    """
    nam = "LevMar: "
    
    n_rejected = 0; too_small = False
    lamb = _checkLambda(init_lambda * 1.0, lmin, lmax, lstep)
    
    
    #
    # Prepare input arrays and check parameter limits5
    #
    x = np.ascontiguousarray(par_in, dtype='float64').flatten()
    o = np.ascontiguousarray(obs_in, dtype='float64').flatten()
    s = np.ascontiguousarray(sig_in, dtype='float64').flatten()
    
    if(not CheckPars(x, pinfo)):
        print("Please fix your input, exiting")
        return None

    #
    # Normalize parameters according to the user Norm
    #
    status = NormalizePars(x, pinfo)


    #
    # Get first evaluation of Chi2 and init Jacobian
    #
    syn, J  = _eval_fx(fx, x, pinfo, udat, auto_derivatives=auto_derivatives, get_J = True)
    res     = _getResidue(syn, o, s, pinfo, J = J)

    bestChi2 = _getChi2(res)
    best_x   = np.copy(x)
    if(verbose):
        print(nam +  "Init Chi2={0}".format(bestChi2))

    
    #
    # Init iterations
    #
    for ii in range(Niter):
        olamb = lamb*1
        
        x = np.copy(best_x)
        
        # Get model correction for current lambda and J

        chi2, xnew, lamb = _getNewEstimate(fx, x, o, s, res, pinfo, udat, lamb, J, lmin, lmax, lstep, auto_derivatives = auto_derivatives, svd_thres=svd_thres)

        
        dchi2 = np.abs((bestChi2 - chi2) / chi2)
        
        # Check if the step improves things
        if(chi2 < bestChi2):
            bestChi2 = chi2*1
            best_x = np.copy(xnew)
            n_rejected = 0

            olamb = lamb*1
            lamb = _checkLambda(lamb/lstep, lmin, lmax, lstep)

            # Is the correction too small for 2 consecutive iterations?
            if(dchi2 < fx_thres):
                if too_small:
                    if(verbose):
                        print(nam+" iter={0}, Chi2={1}, dChi2={2}, lambda={3}".format(ii, bestChi2, dchi2, olamb))
                        print(nam+"terminating, two consecutive small iterations")

                    break
                else:
                    too_small = True
            else:
                too_small = False
            if(verbose):
                print(nam+" iter={0}, Chi2={1}, dChi2={2}, lambda={3}".format(ii, bestChi2, dchi2, olamb))

            # Check if we have reached the chi2_thres indicated by the user
            if(bestChi2 < chi2_thres):
                if(verbose):
                    print(nam+"Chi2_threshold ({0}) reached -> Chi2={1}".format(chi2_thres, bestChi2))
                break
            
            # Update J and res
            x = np.copy(best_x)
            syn, J = _eval_fx(fx, x, pinfo, udat, auto_derivatives=auto_derivatives, get_J = True)
            res    = _getResidue(syn, o, s, pinfo, J = J)
            
        else:
            # If the iteration increases Chi2, then increase the Lambda parameter by lstep**2
            olamb = lamb*1
            lamb = _checkLambda(lamb*lstep*lstep, lmin, lmax, lstep)

            n_rejected += 1
            if(verbose):
                print(nam+"Chi2 > best Chi2 ({2} > {3}), Increasing lambda ({0} -> {1})".format(olamb, lamb, chi2, bestChi2))

            if(n_rejected >= n_reject_max):
                if(verbose):
                    print(nam+"Rejected too many iterations, terminating")
                break
            else:
                continue




    if(verbose):
        print(nam+"Final Chi2={0}".format(bestChi2))


    #
    # Get synthetic and derivatives
    #
    syn, J = _eval_fx(fx, np.copy(best_x), pinfo, udat, auto_derivatives=auto_derivatives, get_J = True)
    status = ScalePars(best_x, pinfo)

    return bestChi2, best_x, syn, J
