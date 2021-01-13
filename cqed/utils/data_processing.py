"""
A set of functions that can be applied to data. Could probably be housed/grouped together elsewhere.

"""

import numpy as np
import scipy as sp

def IQangle(data):
    I = np.real(data)
    Q = np.imag(data)
    Cov = np.cov(I,Q)
    A = sp.linalg.eig(Cov)
    eigvecs = A[1]
    if A[0][1]>A[0][0]:
        eigvec1 = eigvecs[:,0]
    else:
        eigvec1 = eigvecs[:,1]
    theta = np.arctan(eigvec1[0]/eigvec1[1])
    return theta

def IQrotate(data, theta):
    return data*np.exp(1.j*theta)