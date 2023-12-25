import numpy as np
from scipy.optimize import minimize

def lineFreeF(params, x):
    slope, intercept = params
    return slope * np.array(x) + intercept

def lineConstF(params, x):
    intercept = params
    return np.repeat(intercept, len(x))

def lineZeroF(params, x):
    slope = params
    return slope * np.array(x)

def chi2F(params, f, x, y, uy):
    y_pred = f(params, x)
    residuals = (y - y_pred) / uy
    return np.sum(residuals**2)
    
def LRTFreeConst(x, y, uy):

    init_free = [1, 1]
    init_const = [np.mean(y)]
    
    LS_free  = minimize(chi2F, init_free,  args=(lineFreeF,  x, y, uy))
    LS_const = minimize(chi2F, init_const, args=(lineConstF, x, y, uy))
    
    lrt = np.sqrt(LS_const.fun - LS_free.fun) #* (LS_free.fun - LS_const.fun) / np.abs(LS_free.fun - LS_const.fun)

    return lrt, LS_free.x, LS_const.x

def LRTFreeZero(x, y, uy):
    init_free = [1, 1]
    init_zero = [0]
    
    LS_free = minimize(chi2F, init_free, args=(lineFreeF, x, y, uy))
    LS_zero = minimize(chi2F, init_zero, args=(lineZeroF, x, y, uy))
    
    lrt = np.sqrt(LS_zero.fun - LS_free.fun)

    return lrt, LS_free.x, LS_zero.x

def likelihood_plaw(params, x, y):
    A, k = params[0], params[1]
    y_pred = plaw(x, A, k)
    return np.sum((y - y_pred)**2)

def plaw(x, A, k):
    return A * x ** k