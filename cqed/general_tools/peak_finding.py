# Authors: Arno Bargerbos @TUDelft
# 22-APR-2020

"""
A set of functions used to find minima in (potentially noisy) data.

Would be good to identify some other options, but ultimately it will depend on the quality of the data and is thus a bit user dependent.
"""

import numpy as np
from scipy import signal


def find_minimum_absolute(fs, ys):
    """ 
    Finds the frequency belonging to the minimal amplitude of a trace.

    fs (any): array of x values
    ys (any) array of y values
    
    returns:
    value of fs that corresponds to the minimum of ys
    """
    return fs[np.argmin(ys)]


def find_minimum_filtered(fs, ys, window=1e6, prominence=4):

    """ Taking the minimum of a trace to find a resonance is prone to noise. 
    We therefore filter the data based on a savgol filter with a window not 
    smaller than the linewidth of interest. We then look for dips using 
    prominence based peak finding.

    args:
    fs (any): array of x values
    ys (any) array of y values
    window (same as fs): window used to filter data. Should not be larger than resonance linewidth of interest.
    prominence (same as ys): the topographic prominense a dip needs to have to be considered a dip. 
    
    returns:
    value of fs that corresponds to the identified minimum in ys
    Needs testing. 
    """

    min_peakwidth = window / (fs[1] - fs[0])
    yhat = signal.savgol_filter(ys, max(_round_up_to_odd(min_peakwidth * 3), 3), 2) #window length must be odd and >= polyorder
    peaks, _ = signal.find_peaks(-1.0 * yhat, prominence=prominence)

    if len(peaks) == 0:
        resfreq = None
    elif len(peaks) > 1:
        peakarg = np.argmin(yhat[peaks])
        resfreq = fs[peaks[peakarg]]
    else:
        resfreq = fs[peaks][0]

    return resfreq


def _round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)
