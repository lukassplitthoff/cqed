# Authors: Arno Bargerbos @TUDelft
# 22-APR-2020

"""
A set of functions used to find extrema in (potentially noisy) data.

Would be good to identify some other options, but ultimately it will depend on the quality of the data and is thus a bit user dependent.
"""

import numpy as np
from scipy import signal

def _reflect_in_mean(ys):
    """ 
    Function to change a dip into a peak, so that we can also look for peaks in data with dips
    """
    return -ys+2*np.mean(ys)

def _round_up_to_odd(f):
    return int(np.ceil(f) // 2 * 2 + 1)

def _convert_to_dBm(ys):
    return 20 * np.log10(ys)

def find_peak_literal(fs, ys, reflect=False, dBm=False):
    """ 
    Finds the frequency belonging to the literal maximal amplitude of a trace.

    fs (any): array of x values
    ys (any) array of y values
    
    returns:
    value of fs that corresponds to the maximum of ys
    """

    if reflect==True:
        ys = _reflect_in_mean(ys)
    if dBm==True:
        ys = _convert_to_dBm(ys)

    return fs[np.argmax(ys)]


def find_peak_filtered(fs, ys, window=1e6, prominence=4):

    """ Taking the maximum of a trace to find a resonance is prone to noise. 
    We therefore filter the data based on a savgol filter with a window not 
    smaller than the linewidth of interest. We then look for peaks using 
    prominence based peak finding.

    args:
    fs (any): array of x values
    ys (any) array of y values
    window (same as fs): window used to filter data. Should not be larger than resonance linewidth of interest.
    prominence (same as ys): the topographic prominense a peak needs to have to be considered a peak. 
    
    returns:
    value of fs that corresponds to the identified maximum in ys
    Needs testing. 
    """

    min_peakwidth = window / (fs[1] - fs[0])
    yhat = signal.savgol_filter(ys, max(_round_up_to_odd(min_peakwidth * 3), 3), 2) #window length must be odd and >= polyorder
    peaks, _ = signal.find_peaks(yhat, prominence=prominence)

    if len(peaks) == 0:
        resfreq = None
    elif len(peaks) > 1:
        peakarg = np.argmax(yhat[peaks])
        resfreq = fs[peaks[peakarg]]
    else:
        resfreq = fs[peaks][0]

    return resfreq
