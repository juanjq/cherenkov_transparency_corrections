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

def likelihood_plaw(params, x, y, reference_intensity):
    """
    Calculate the likelihood of a power-law model given the parameters and observed data.

    Parameters:
        params (list): List of parameters [A, k] for the power-law model.
        x (array-like): Array of x-values.
        y (array-like): Array of observed y-values.

    Returns:
        float: The likelihood value.

    """
    norm, pindex = params[0], params[1]
    y_pred = powerlaw(x, norm, pindex)
    return np.sum((y - y_pred)**2)

def powerlaw(x, norm, pindex):
    """
    Power-law function.

    Parameters:
    x (float): The input value.
    norm (float): The amplitude.
    pindex (float): The power-law exponent.

    Returns:
    float: The calculated value of the power-law function.
    """
    return norm * (x) ** pindex

    
def expfunc(x, a, b):
    """
    Exponential function.

    Parameters:
    x (float): The input value.
    a (float): The amplitude.
    b (float): The exponential index.

    Returns:
    float: The calculated value of the exponential function.
    """    
    return a * np.exp(b * x)


def calc_light_yield(normr, normf, alphaf):
    """
    Calculate the light yield based on the reference point, AR, alphaR, A2, and alpha2.

    Parameters:
    refpoint (float): The reference point.
    AR (float): The AR value.
    alphaR (float): The alphaR value.
    A2 (float): The A2 value.
    alpha2 (float): The alpha2 value.

    Returns:
    float: The calculated light yield.
    """
    return (normr / normf) ** (1 / ( 1 + alphaf))
    
def straight_line(x, intercept, slope):
    """
    Straight line function.

    Parameters:
    x (float): The input value.
    intercept (float): The intercept.
    slope (float): The slope.

    Returns:
    float: The calculated value of the straight line function.
    """
    return intercept + slope * x
    
def pol2(x, a, b, c):
    """
    Calculates the value of a second-degree polynomial.

    Parameters:
    x (float): The input value.
    a (float): The coefficient of the constant term.
    b (float): The coefficient of the linear term.
    c (float): The coefficient of the quadratic term.

    Returns:
    float: The value of the polynomial at the given input value.
    """
    return a + b * x + c * x * x

def angular_dist(az1, az2):
    """
    Calculate the angular distance between two azimuth angles.

    Parameters:
    az1 (float): The first azimuth angle in degrees.
    az2 (float): The second azimuth angle in degrees.

    Returns:
    float: The angular distance between the two azimuth angles.
    """
    angular_distance_abs = abs(az1 - az2)
    return min(angular_distance_abs, 360 - angular_distance_abs)