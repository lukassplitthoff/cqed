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
    freqs = np.linspace(station.vna.S21.start(), station.vna.S21.stop(), station.vna.S21.npts())

    if not station.vna.rf_power():
        station.vna.rf_on()

    vna_data = station.vna.S21.trace_mag_phase()

    return [freqs, vna_data[0], vna_data[1]]


def transmission_vs_frequency_explicit(center, span, suffix=None):
    @MakeMeasurementFunction([DataParameter(name='frequency' + str(suffix),
                                            unit='Hz',
                                            paramtype='array',
                                            # yes independent, but pysweep will not recognize it as such
                                            independent=2
                                            ),
                              DataParameter(name='amplitude' + str(suffix),
                                            unit='',
                                            paramtype='array',
                                            # explicitely tell that this parameter depends on
                                            # the corresponding frequency parameter
                                            extra_dependencies=['frequency' + str(suffix)]
                                            ),
                              DataParameter(name='phase' + str(suffix),
                                            unit='rad',
                                            paramtype='array',
                                            # explicitely tell that this parameter depends on
                                            # the corresponding frequency parameter
                                            extra_dependencies=['frequency' + str(suffix)]
                                            )
                              ])
    def transmission_vs_frequency_measurement_function(d):
        station = d['STATION']
        station.vna.S21.center(center)
        station.vna.S21.span(span)
        return return_vna_trace(d)

    # return a measurement function
    return transmission_vs_frequency_measurement_function


def multiple_meas_functions(freq_list, span_list):
    fun_str = ''

    for i, c in enumerate(freq_list):
        s = span_list[i]
        fun_str += 'cvna.transmission_vs_frequency_explicit({}, {}, suffix={})+'.format(c, s, i)
    fun_str = fun_str[:-1]
    return fun_str