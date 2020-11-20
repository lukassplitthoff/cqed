"""
A set of functions to conveniently use for Alazar measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np
from pytopo.rf.alazar.softsweep import setup_triggered_softsweep
from pytopo.rf.alazar.awg_sequences import TriggerSequence
import qcodes

def setup_single_averaged_IQpoint(controller, time_bin, integration_time, setup_awg=True,
                                  post_integration_delay=10e-6,
                                  verbose=True, allocated_buffers=None):
    """
    Setup the alazar to measure a single IQ value / buffer.

    Note: we always average over buffers here (determined by time_bin and integration_time).
    This implies that you need to use a trigger sequence with a trigger interval that 
    corresponds to an even number of IF periods.
    """
    station = qcodes.Station.default
    alazar = station.alazar

    navgs = int(integration_time / time_bin)

    if setup_awg:
        trig_seq = TriggerSequence(station.awg, SR=1e7)
        trig_seq.wait = 'off'
        trig_seq.setup_awg(
            cycle_time=time_bin, debug_signal=False, ncycles=1, plot=False)

    ctl = controller
    ctl.buffers_per_block(None)
    ctl.average_buffers(True)

    ctl.setup_acquisition(samples=int((time_bin-post_integration_delay) * alazar.sample_rate() // 128 * 128),
                          records=1, buffers=navgs, allocated_buffers=allocated_buffers, verbose=verbose)

def measure_single_averaged_IQpoint(controller, time_bin, integration_time, channel=0, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("amplitude", "", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_point(d):   
        # setup_single_averaged_IQpoint(controller, time_bin, integration_time, setup_awg=True,
        #                           verbose=True, allocated_buffers=None)

        station = d["STATION"]
        data = controller.acquisition()
        data = np.squeeze(data)[..., 0].mean()
        mag, phase = np.abs(data), np.angle(data, deg=False)

        return [mag, phase]
    return return_alazar_point

def measure_soft_time_avg_spec(controller, sweep_param, sweep_vals, integration_time, 
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

def measure_soft_time_avg_spec_optimized(controller, sweep_param, sweep_vals, integration_time, hetsrc, hetsrc_freq, 
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
        cal = measure_soft_time_avg_spec(controller, sweep_param, sweep_vals, integration_time, 
                                exp_name=None, **kw)(d)
        m0 = res_finder(cal[0], 20 * np.log10(cal[1]))
        if m0 == None:
            raise Exception(
                "Failed to find a resonance."
            )  # needs work, can implement alternative strategies
        d["f0"] = m0
        return [m0]

    return resonance_estimate_measurement_function