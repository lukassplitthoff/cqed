""" Set of generic functions used for fitting """
import numpy as np


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
