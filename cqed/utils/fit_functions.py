""" Set of generic functions used for fitting """
import numpy as np
from scipy.signal import argrelextrema, savgol_filter


def dbl_gaussian(x, c1, mu1, sg1, c2, mu2, sg2):
    """
    A double gaussian distribution
    @param x: x-axis
    @param c1: scaling parameter for distribution 1
    @param mu1: mean of first gaussian
    @param sg1: variance of first gaussian
    @param c2: scaling parameter for distribution 2
    @param mu2: mean of second gaussian
    @param sg2: variance of second gaussian
    @return: array of double gaussian distribution
    """
    res = c1 * np.exp(-(x - mu1) ** 2. / (2. * sg1 ** 2.)) \
        + c2 * np.exp(-(x - mu2) ** 2. / (2. * sg2 ** 2.))
    return res


def dbl_gaussian_guess_means(xdat, ydat, threshold=0.1):
    '''
    Calculates initial guesses for the two means for a double Gaussian fit.
    After smoothing the data, it first finds a local minimum to split the data
    in two halfs, with a peak in each half. It then finds the maximum at each
    side of the local minimum to estimate x1 and x2, the centers of the two
    Gaussian peaks. If there is no central local minimum, it finds the maximum
    of the data and returns it as the guess for both peaks.
    '''

    M = np.max(ydat)
    m = len(ydat)
    m1 = 0
    while ydat[m1] < threshold * M:
        m1 += 1
    m2 = m - 1
    while ydat[m2] < threshold * M:
        m2 -= 1

    n = 2 * int(0.1 * (m2 - m1)) + 1
    y_smoothed = savgol_filter(ydat[m1:m2], n, 3)
    y_smoothed = savgol_filter(y_smoothed, n, 3)
    y_smoothed = savgol_filter(y_smoothed, n, 3)
    ii = np.array(argrelextrema(y_smoothed, np.less)) + m1
    ii = ii[0]
    if len(ii) == 0:
        # If there is no central minimum
        ix1 = np.argmax(ydat)
        ix2 = ix1
    elif len(ii) == 1:
        # If there is just one local minimum (ideal case)
        i = ii[0]
        ix1 = np.argmax(ydat[0:i])
        ix2 = i + np.argmax(ydat[i:m - 1])
    else:
        # If it found more than one local minimum
        i = np.min(ii)
        ix1 = np.argmax(ydat[0:i])
        ix2 = i + np.argmax(ydat[i:m - 1])

    return xdat[ix1], xdat[ix2]


def gaussian_guess_sigma_A(xdat, ydat, threshold=0.2):
    '''
    Calculates some initial guess for
    sigma and A for a double Gaussian fit.
    '''

    M = np.max(ydat)
    m = len(ydat)
    m1 = 0
    while ydat[m1]<threshold*M:
        m1+=1
    m2 = m-1
    while ydat[m2]<threshold*M:
        m2-=1

    sigma_guess = 0.15*(xdat[m2]-xdat[m1])
    y_max = ydat[np.argmax(ydat)]
    A_guess = 0.5 * y_max * np.sqrt(2*np.pi*sigma_guess**2)

    return sigma_guess, A_guess


def exp_func(x, gamma, a):
    """ Exponential function a * exp(gamma * x)
    @param x: array
    @param gamma: rate
    @param a: scaling prefactor
    @return: array
    """
    return a * np.exp(gamma * x)


def lin_func(x, m, c):
    """A simple linear function m*x+c
    @param x: x-axis
    @param m: slope
    @param c: offset"""

    return m * x + c


def QPP_Lorentzian(f, params):
    '''
    Lorentzian function.
    '''
    Gamma = params["Gamma"]
    a = params["a"]
    b = params["b"]

    return a * (4*Gamma / ((2*Gamma)**2+(2*np.pi*f)**2)) + b


def residual(params, x, y, function):
    y_model = function(x, params)
    return y_model - y
