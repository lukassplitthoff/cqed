"""
A set of functions to conveniently use for Alazar measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np
from pytopo.rf.alazar.softsweep import setup_triggered_softsweep

def measure_trace_alazar(controller, sweep_param, sweep_vals, integration_time, 
                                exp_name=None, channel=0, **kw):


    @MakeMeasurementFunction(
        [
            DataParameter("frequency", "Hz", "array", True),
            DataParameter("amplitude", "", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):
        setup_triggered_softsweep(controller, sweep_param, sweep_vals, integration_time,
                                  setup_awg=True, verbose=True, **kw)      
        station = d["STATION"]
        freqs = sweep_vals

        data = np.squeeze(controller.acquisition())[..., channel]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        return [freqs, mag, phase]
    return return_alazar_trace

def measure_trace_alazar_optimized(controller, sweep_param, sweep_vals, integration_time, hetsrc, hetsrc_freq, 
                                exp_name=None, channel=0, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("frequency", "Hz", "array", True),
            DataParameter("amplitude", "", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):

        station = d["STATION"]
        if "f0" not in d:
            d["f0"] = hetsrc_freq
            
        hetsrc.frequency(d["f0"])
        setup_triggered_softsweep(controller, sweep_param, sweep_vals, integration_time,
                                  setup_awg=True, verbose=True, **kw)      
        freqs = sweep_vals

        data = np.squeeze(controller.acquisition())[..., channel]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        return [freqs, mag, phase]
    return return_alazar_trace


def measure_resonance_estimate(controller, sweep_param, sweep_vals, integration_time, res_finder, f0,
                                  setup_awg=True, verbose=True, **kw):
    """Pysweep VNA measurement function that measures an estimated resonance
    frequency.

    The idea is to use this over large ranges to figure out the approximate
    resonance frequency and then set up a finer scan. Needs testing.
    f0 (Hz): Frequency around which to measure. Ignored if there is already an f0 in d['f0'].
    fspan (Hz): Span around f0 to measure.
    fstep (Hz): Frequency step size.
    res finder: Function that finds a resonance from VNA output. WIP.
    kwargs: see `setup_frq_sweep`.

    Returns:
    Pysweep measurement function

    Returns:
    Pysweep measurement function

    """

    @MakeMeasurementFunction([DataParameter(name="resonance_frequency", unit="Hz")])
    def resonance_estimate_measurement_function(d):
        if "f0" not in d:
            d["f0"] = f0
        station = d["STATION"]
        cal = measure_trace_alazar(controller, sweep_param, sweep_vals, integration_time, 
                                exp_name=None, **kw)(d)
        m0 = res_finder(cal[0], 20 * np.log10(cal[1]))
        if m0 == None:
            raise Exception(
                "Failed to find a resonance."
            )  # needs work, can implement alternative strategies
        d["f0"] = m0
        return [m0]

    return resonance_estimate_measurement_function