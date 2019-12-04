# Authors: Lukas Splitthoff, Lukas Gruenhaupt @TUDelft
# 04-DEC-2019

''' A set of functions to conveniently use for VNA measurements with pysweep.'''

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np

@MakeMeasurementFunction([DataParameter('frequency','Hz', 'array', True),
                          DataParameter('amplitude', '', 'array'), 
                          DataParameter('phase', 'rad', 'array')])
def return_vna_trace(d):
    """
    Pysweep VNA measurement function.
    Returns VNA frequency axis, linear amplitude and phase in radians.
    Keeps currently set VNA paramaters.
    """
    station = d['STATION']
    freqs = np.linspace(station.vna.S21.start(),station.vna.S21.stop(), station.vna.S21.npts())
    if not station.vna.rf_power():
        station.vna.rf_on()
    vna_data = station.vna.S21.trace_mag_phase()
    return [freqs, vna_data[0], vna_data[1]]