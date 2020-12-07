"""
A set of functions to conveniently use for Alazar measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np
from pytopo.rf.alazar.softsweep import setup_triggered_softsweep
from pytopo.rf.alazar.awg_sequences import TriggerSequence
import cqed.awg_sequences
import qcodes
from cqed.awg_sequences.awg_sequences import RabiSequence, RamseySequence, T1Sequence, EchoSequence, QPTriggerSequence
import time

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
    ctl.average_buffers(None)
    ctl.average_buffers_postdemod(True)
    
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

def setup_time_rabi(controller, pulse_times, readout_time, 
                              navgs=500, acq_time=2.56e-6, setup_awg=True):
    """
    Set up ...
    """
    
    station = qcodes.Station.default
    
    # setting up the AWG
    if setup_awg:
        seq = RabiSequence(station.awg, SR=1e9)
        seq.wait = 'all'
        seq.setup_awg(pulse_times=pulse_times, readout_time=readout_time, cycle_time=20e-6, start_awg=True)
        
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)  
    controller.setup_acquisition(samples=None, records=pulse_times.size, buffers=navgs, acq_time=acq_time, verbose=False)

def measure_time_rabi(controller, pulse_times, readout_time,  
                              navgs=500, acq_time=2.56e-6, setup_awg=True, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("pulse_time", "s", "array", True),
            DataParameter("amplitude", "V", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):

        setup_time_rabi(controller, pulse_times, readout_time, navgs=navgs, acq_time=acq_time,
                                  setup_awg=setup_awg, **kw)      

        times = pulse_times
        station = d["STATION"]
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()
        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)
        station.fg.ch1.state('OFF')
        station.awg.stop()
        time.sleep(0.1)

        #it is unclear which combination of off and stop and sleep is required
        #but without them the timing goes wrong

        return [times, mag, phase]
    return return_alazar_trace

def setup_ramsey(controller, delays, pulse_time, readout_time, 
                              navgs=500, acq_time=2.56e-6, setup_awg=True):
    """
    Set up ...
    """
    
    station = qcodes.Station.default
    
    # setting up the AWG
    if setup_awg:
        seq = RamseySequence(station.awg, SR=1e9)
        seq.wait = 'all'
        seq.setup_awg(delays = delays, pulse_time=pulse_time, readout_time=readout_time, cycle_time = 20e-6, start_awg=True)
        
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)  
    controller.setup_acquisition(samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)

def measure_ramsey(controller, delays, pulse_time, readout_time,  
                              navgs=500, acq_time=2.56e-6, setup_awg=True, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("delay", "s", "array", True),
            DataParameter("amplitude", "V", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):

        setup_ramsey(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                                  setup_awg=setup_awg, **kw)      

        times = delays
        station = d["STATION"]
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()
        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)
        station.fg.ch1.state('OFF')
        station.awg.stop()
        time.sleep(0.1)

        #it is unclear which combination of off and stop and sleep is required
        #but without them the timing goes wrong

        return [times, mag, phase]
    return return_alazar_trace


def setup_T1(controller, delays, pulse_time, readout_time, 
                              navgs=500, acq_time=2.56e-6, setup_awg=True):
    """
    Set up ...
    """
    
    station = qcodes.Station.default
    
    # setting up the AWG
    if setup_awg:
        seq = T1Sequence(station.awg, SR=1e9)
        seq.wait = 'all'
        seq.setup_awg(delays = delays, pulse_time=pulse_time, readout_time=readout_time, cycle_time = 20e-6, start_awg=True)
        
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)  
    controller.setup_acquisition(samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)

def measure_T1(controller, delays, pulse_time, readout_time,  
                              navgs=500, acq_time=2.56e-6, setup_awg=True, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("delay", "s", "array", True),
            DataParameter("amplitude", "V", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):

        setup_T1(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                                  setup_awg=setup_awg, **kw)      

        times = delays
        station = d["STATION"]
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()
        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)
        station.fg.ch1.state('OFF')
        station.awg.stop()
        time.sleep(0.1)

        #it is unclear which combination of off and stop and sleep is required
        #but without them the timing goes wrong

        return [times, mag, phase]
    return return_alazar_trace


def setup_echo(controller, delays, pulse_time, readout_time, 
                              navgs=500, acq_time=2.56e-6, setup_awg=True):
    """
    Set up ...
    """
    
    station = qcodes.Station.default
    
    # setting up the AWG
    if setup_awg:
        seq = EchoSequence(station.awg, SR=1e9)
        seq.wait = 'all'
        seq.setup_awg(delays = delays, pulse_time=pulse_time, readout_time=readout_time, cycle_time = 20e-6, start_awg=True)
        
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)  
    controller.setup_acquisition(samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)

def measure_echo(controller, delays, pulse_time, readout_time,  
                              navgs=500, acq_time=2.56e-6, setup_awg=True, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("delay", "s", "array", True),
            DataParameter("amplitude", "V", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def return_alazar_trace(d):

        setup_echo(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                                  setup_awg=setup_awg, **kw)      

        times = delays
        station = d["STATION"]
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()
        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)
        station.fg.ch1.state('OFF')
        station.awg.stop()
        time.sleep(0.1)

        #it is unclear which combination of off and stop and sleep is required
        #but without them the timing goes wrong

        return [times, mag, phase]
    return return_alazar_trace


def setup_QPP(controller, acq_time, navg, SR=250e6, setup_awg=True):
    """
    Set up ...
    """

    station = qcodes.Station.default

    # setting up the AWG
    if setup_awg:
        seq = QPTriggerSequence(station.awg, SR=1e7)
        seq.load_sequence(cycle_time=acq_time+1e-3, plot=False, use_event_seq = True, ncycles = navg)
        
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(False)  
    station.alazar.sample_rate(int(SR))
    npoints = int(acq_time*SR // 128 * 128)
    controller.setup_acquisition(npoints, 1, navg)
    print(controller, navg, acq_time, controller.demod_frq(), npoints)

def measure_QPP(controller, acq_time, navg, SR=250e6, setup_awg=True, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("timestamp", "s", "array", True),
        ]
    )
    def return_alazar_trace(d):

        setup_QPP(controller, acq_time, navg, SR=SR, setup_awg=setup_awg, **kw)      

        station = d["STATION"]
        station.alazar.clear_buffers()
        data = np.squeeze(controller.acquisition())[...,0]
        time.sleep(0.1)

        timestamp = int(time.time()*1e6)
        datasaver_run_id = d["DATASAVER"].datasaver._dataset.run_id
        data_folder_path = str(qcodes.config.core.db_location)[:-3]+"\\"
        np.save(data_folder_path+"ID_"+f"{datasaver_run_id}_IQ_{timestamp:d}",[controller.demod_tvals, data])

        return [timestamp]
    return return_alazar_trace