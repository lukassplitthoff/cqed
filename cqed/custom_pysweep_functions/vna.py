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
    Takes care of vna timeout automatically, by running the measurement in a context manager with properly set timeout.
    """
    station = d['STATION']
    freqs = np.linspace(station.vna.S21.start(),station.vna.S21.stop(), station.vna.S21.npts())

    if not station.vna.rf_power():
        station.vna.rf_on()

    # Set timeout long enough for measurement only during the measurement.
    # This approach keeps the timeout short for standard operations, to avoid unneccessary long waiting times.
    # Ideally, this should be taken care of in the instrument driver.
    
    try:
        #If parameter is implemented, get sweep time directly from VNA, add 1s for good measure.
        timeout = station.vna.S21.sweep_time() + 1 
    except AttributeError:
        #If parameter is not implemented, estimate from npts and bandwidth and add 5s to be on the safe side.
        timeout = station.vna.S21.npts()/station.vna.S21.bandwidth() + 5 

    with station.vna.timeout.set_to(timeout):
        vna_data = station.vna.S21.trace_mag_phase()

    return [freqs, vna_data[0], vna_data[1]]