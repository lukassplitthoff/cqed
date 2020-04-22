# Authors: Arno Bargerbos @TUDelft
# 22-APR-2020

"""
A set of functions used to find minima in (potentially noisy) data.
"""

import numpy as np
from scipy import signal


def find_minimum_absolute(fs, ys):
    """ 
    Finds the frequency belonging to the minimal amplitude of a trace.
    """
    return fs[np.argmin(ys)]


def find_minimum_filtered(fs, ys, min_linewidth=1e6, min_depth=4):

    """ Taking the minimum of a trace to find a resonance is prone to noise. 
    Here we smooth the data with a window based on the expected linewidth of the dip, 
    and look for dips that are sufficiently deep based on the (topographic) prominence.
    
    Needs testing. 
    """

    min_peakwidth = min_linewidth / (fs[1] - fs[0])
    yhat = signal.savgol_filter(ys, max(_round_up_to_odd(min_peakwidth * 3), 3), 2)
    peaks, _ = signal.find_peaks(-1.0 * yhat, prominence=min_depth)

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

