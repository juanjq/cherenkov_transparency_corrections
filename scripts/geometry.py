import numpy as np
from scipy.optimize import minimize

def lineFreeF(params, x):
    """
    Calculate the y-values of a line given the slope and intercept.

    Parameters:
    params (tuple): A tuple containing the slope and intercept of the line.
    x (list or array-like): The x-values at which to calculate the y-values.

    Returns:
    numpy.ndarray: The calculated y-values of the line.
    """
    slope, intercept = params
    return slope * np.array(x) + intercept

def lineConstF(params, x):
    """
    Returns an array of the same length as x, where each element is equal to the intercept value.

    Parameters:
    - params: The intercept value.
    - x: The input array.

    Returns:
    - An array of the same length as x, where each element is equal to the intercept value.
    """
    intercept = params
    return np.repeat(intercept, len(x))

def lineZeroF(params, x):
    """
    Calculate the y-coordinate of a point on a line with zero y-intercept.

    Parameters:
    params (float): The slope of the line.
    x (float): The x-coordinate of the point.

    Returns:
    float: The y-coordinate of the point on the line.
    """
    slope = params
    return slope * np.array(x)

def chi2F(params, f, x, y, uy):
    """
    Calculate the chi-squared value for a given set of parameters.

    Parameters:
        params (array-like): The parameters to be used in the function `f`.
        f (callable): The function to be evaluated.
        x (array-like): The independent variable values.
        y (array-like): The observed dependent variable values.
        uy (array-like): The uncertainties in the observed dependent variable values.

    Returns:
        float: The chi-squared value.
    """
    y_pred = f(params, x)
    residuals = (y - y_pred) / uy
    return np.sum(residuals**2)
    
def LRTFreeConst(x, y, uy):
    """
    Calculates the likelihood ratio test (LRT) statistic for the comparison of a free model and a constrained model.

    Parameters:
    x (array-like): The x-coordinates of the data points.
    y (array-like): The y-coordinates of the data points.
    uy (array-like): The uncertainties of the y-coordinates.

    Returns:
    lrt (float): The LRT statistic.
    free_params (array-like): The parameter values of the free model that minimize the chi-square function.
    const_params (array-like): The parameter values of the constrained model that minimize the chi-square function.
    """
    init_free = [1, 1]
    init_const = [np.mean(y)]
    
    LS_free  = minimize(chi2F, init_free,  args=(lineFreeF,  x, y, uy))
    LS_const = minimize(chi2F, init_const, args=(lineConstF, x, y, uy))
    
    lrt = np.sqrt(LS_const.fun - LS_free.fun) #* (LS_free.fun - LS_const.fun) / np.abs(LS_free.fun - LS_const.fun)

    return lrt, LS_free.x, LS_const.x

def LRTFreeZero(x, y, uy):
    """
    Calculates the likelihood ratio test (LRT) statistic for the hypothesis of a free model versus a zero model.
    
    Parameters:
    - x: array-like, x-coordinates of the data points
    - y: array-like, y-coordinates of the data points
    - uy: array-like, uncertainties of the y-coordinates
    
    Returns:
    - lrt: float, the LRT statistic
    - LS_free.x: array-like, the parameter values that minimize the chi-square function for the free model
    - LS_zero.x: array-like, the parameter values that minimize the chi-square function for the zero model
    """
    init_free = [1, 1]
    init_zero = [0]
    
    LS_free = minimize(chi2F, init_free, args=(lineFreeF, x, y, uy))
    LS_zero = minimize(chi2F, init_zero, args=(lineZeroF, x, y, uy))
    
    lrt = np.sqrt(LS_zero.fun - LS_free.fun)

    return lrt, LS_free.x, LS_zero.x

def likelihood_plaw(params, x, y):
    """
    Calculate the likelihood of a power-law model given the parameters and observed data.

    Parameters:
        params (list): List of parameters [A, k] for the power-law model.
        x (array-like): Array of x-values.
        y (array-like): Array of observed y-values.

    Returns:
        float: The likelihood value.

    """
    A, k = params[0], params[1]
    y_pred = plaw(x, A, k)
    return np.sum((y - y_pred)**2)

def plaw(x, A, k):
    """
    Calculate the power-law function.

    Parameters:
    x (float): The input value.
    A (float): The amplitude of the power-law function.
    k (float): The exponent of the power-law function.

    Returns:
    float: The result of the power-law function.
    """
    return A * x ** k