"""
A set of functions to conveniently use for Alazar measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction, MeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np
from pytopo.rf.alazar.softsweep import setup_triggered_softsweep
from pytopo.rf.alazar.awg_sequences import TriggerSequence
import cqed.awg_sequences
import qcodes
from cqed.awg_sequences.awg_sequences import RabiSequence, RamseySequence, T1Sequence, EchoSequence, QPTriggerSequence
import time
from pathlib import Path
import lmfit
import cqed.utils.data_processing as dp
from scipy.signal import savgol_filter


def setup_time_rabi(controller, pulse_times, readout_time,
                    navgs=500, acq_time=2.56e-6):
    """Function that sets up a Tektronix AWG5014C sequence as well as the Alazar controller for 
    performing a time Rabi measurement.

    Args:
        controller (QCoDeS instrument): alazar controller for handling acquisition
        pulse_times (array, s): pulse times for which the Rabi sequence will be performed
        readout_time (s): time during which the readout tone will be on
        navgs (int): number of times each rabi sequence is performed and then averaged
        acq_time (s): I have forgotted what that is

    """

    station = qcodes.Station.default

    # setting up the AWG
    seq = RabiSequence(station.awg, SR=1e9)
    seq.wait = 'all'
    seq.setup_awg(pulse_times=pulse_times, readout_time=readout_time,
                  cycle_time=20e-6, start_awg=True)
    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)
    controller.setup_acquisition(
        samples=None, records=pulse_times.size, buffers=navgs, acq_time=acq_time, verbose=False)


def measure_time_rabi(controller, pulse_times, setup_awg=False, qubsrc_power=None, qubsrc_freq=None, hetsrc_power=None, hetsrc_freq=None, suffix='', fit=False, T1_guess=None, **kw):

    def return_alazar_trace(d):

        if setup_awg:
            setup_time_rabi(controller=controller,
                            pulse_times=pulse_times, **kw)

        times = pulse_times
        station = d["STATION"]

        if qubsrc_power is not None:
            station.qubsrc.power(qubsrc_power)
        if hetsrc_power is not None:
            station.hetsrc.RF.power(hetsrc_power)

        if qubsrc_freq == 'dict':
            station.qubsrc.frequency(d["fq"])
        elif qubsrc_freq == None:
            pass
        else:
            station.qubsrc.frequency(qubsrc_freq)

        if hetsrc_freq == 'dict':
            station.hetsrc.frequency(d["f0"])
        elif hetsrc_freq == None:
            pass
        else:
            station.hetsrc.frequency(hetsrc_freq)

        station.qubsrc.modulation_rf('ON')
        station.qubsrc.output_rf('ON')
        station.RF.on()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('ON')
        station.RF.ref_LO_out('LO')
        station.LO.on()
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()

        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.qubsrc.modulation_rf('OFF')
        station.qubsrc.output_rf('OFF')
        station.RF.off()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('OFF')
        station.RF.ref_LO_out('OFF')
        station.LO.off()
        time.sleep(0.1)

        if fit:

            angle = dp.IQangle(mag*np.exp(1.j*phase))
            rotated_data = dp.IQrotate(mag*np.exp(1.j*phase), angle)
            mod = lmfit.models.ExpressionModel(
                'off + amp * exp(-x/t1) * cos(2*pi/period*x)')
            xdat = times
            ydat = np.real(rotated_data)
            period_estimate = 2 * \
                xdat[np.argmax(np.abs(savgol_filter(ydat, 15, 3)-ydat[0]))]
            off_estimate = np.mean(ydat)
            print("0.5*period_estimate = ", period_estimate*0.5*1e9)
            if ydat[0] > off_estimate:
                A_estimate = np.max(ydat) - np.min(ydat)
            else:
                A_estimate = np.min(ydat) - np.max(ydat)
            if T1_guess == None:
                T1_estimate = 1.5e-6
            elif T1_guess == 'dict':
                if "t1" in list(d.keys()) and d["t1"] < 40e-6:
                    T1_estimate = d["t1"]
                else:
                    print("Not in dict")
                    T1_estimate = 1.5e-6
                print("Using T1 = {} us for the Rabi".format(1e6*T1_estimate))
            else:
                T1_estimate = T1_guess
            params = mod.make_params(
                off=off_estimate, amp=A_estimate, t1=T1_estimate, period=period_estimate)
            params['t1'].set(min=1e-9)
            params['t1'].set(max=50e-6)
            params['period'].set(min=1e-9)
            params['period'].set(max=5e-6)
            out = mod.fit(ydat, params, x=xdat)
            pipulse_time = 1*out.params['period'].value/2
            if pipulse_time < 10e-9:
                pipulse_time = 3*out.params['period'].value/2
            d['pipulse'] = pipulse_time
            d['t1'] = out.params['t1']
            print("pi_pulse = ", pipulse_time*1e9)
            return [times, mag, phase, pipulse_time]

        else:
            return [times, mag, phase]

    if fit == True:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="pulse_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["pulse_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["pulse_time" + str(suffix)],
                          ),
            DataParameter(name="pipulse_length" + str(suffix),
                          unit="s",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="pulse_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["pulse_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["pulse_time" + str(suffix)],
                          ),
        ])


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
        seq.setup_awg(delays=delays, pulse_time=pulse_time,
                      readout_time=readout_time, cycle_time=20e-6, start_awg=True)

    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)
    controller.setup_acquisition(
        samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)


def measure_ramsey(controller, delays, pulse_time, readout_time, qubsrc_power=None, qubsrc_freq=None, hetsrc_power=None, hetsrc_freq=None,
                   navgs=500, acq_time=2.56e-6, setup_awg=True, suffix='', fit=False, **kw):

    def return_alazar_trace(d):
        station = d["STATION"]
        times = delays

        if pulse_time == 'dict':
            setup_ramsey(controller, delays, d['pipulse']/2, readout_time, navgs=navgs, acq_time=acq_time,
                         setup_awg=True, **kw)
        else:
            setup_ramsey(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                         setup_awg=setup_awg, **kw)

        if qubsrc_power is not None:
            station.qubsrc.power(qubsrc_power)
        if hetsrc_power is not None:
            station.hetsrc.RF.power(hetsrc_power)

        if qubsrc_freq == 'dict':
            station.qubsrc.frequency(d["fq"])
        elif qubsrc_freq == None:
            pass
        else:
            station.qubsrc.frequency(qubsrc_freq)

        if hetsrc_freq == 'dict':
            station.hetsrc.frequency(d["f0"])
        elif hetsrc_freq == None:
            pass
        else:
            station.hetsrc.frequency(hetsrc_freq)

        station.qubsrc.modulation_rf('ON')
        station.qubsrc.output_rf('ON')
        station.RF.on()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('ON')
        station.RF.ref_LO_out('LO')
        station.LO.on()
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()

        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.qubsrc.modulation_rf('OFF')
        station.qubsrc.output_rf('OFF')
        station.RF.off()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('OFF')
        station.RF.ref_LO_out('OFF')
        station.LO.off()
        time.sleep(0.1)

        # it is unclear which combination of off and stop and sleep is required
        # but without them the timing goes wrong

        if fit:
            # rotate IQ data
            angle = dp.IQangle(mag*np.exp(1.j*phase))
            rotated_data = dp.IQrotate(mag*np.exp(1.j*phase), angle)

            # fit Rabi; still needs work because it can be off by a factor of pi in the phase depending on the sign of the first value!
            mod = lmfit.models.ExpressionModel(
                'off + amp * exp(-x/t1)*sin(2*pi/period*(x + phase))')
            xdat = times
            ydat = np.real(rotated_data)
            period_estimate = 2 * \
                xdat[np.argmax(np.abs(savgol_filter(ydat, 15, 3)-ydat[0]))]
            # period_estimate = 4*xdat[np.argmax(np.abs(xdat-xdat[0]))]
            params = mod.make_params(off=np.mean(
                ydat), amp=ydat[0], t1=0.15e-6, period=period_estimate, phase=0)
            params['t1'].set(min=1e-9)
            out = mod.fit(ydat, params, x=xdat)
            T2_time = out.params['t1'].value

            return [times, mag, phase, T2_time]

        else:
            return [times, mag, phase]

    if fit == True:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="T2ramsey" + str(suffix),
                          unit="s",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
        ])


def setup_T1(controller, delays, pulse_time, readout_time, navgs=500, acq_time=2.56e-6, setup_awg=True):
    """
    Set up ...
    """

    station = qcodes.Station.default

    # setting up the AWG
    if setup_awg:
        seq = T1Sequence(station.awg, SR=1e9)
        seq.wait = 'all'
        # if
        #     seq.setup_awg(delays = delays, pulse_time=pulse_time, readout_time=readout_time, cycle_time = 20e-6, start_awg=True)
        # else:
        seq.setup_awg(delays=delays, pulse_time=pulse_time,
                      readout_time=readout_time, cycle_time=20e-6, start_awg=True)

    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)
    controller.setup_acquisition(
        samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)


def measure_T1(controller, delays, pulse_time, readout_time, qubsrc_power=None, qubsrc_freq=None, hetsrc_power=None, hetsrc_freq=None,
               navgs=500, acq_time=2.56e-6, setup_awg=True, suffix='', fit=False, T1_guess=None, **kw):

    def return_alazar_trace(d):
        station = d["STATION"]
        times = delays
        if pulse_time == 'dict':
            setup_T1(controller, delays, d['pipulse'], readout_time, navgs=navgs, acq_time=acq_time,
                     setup_awg=True, **kw)
        else:
            setup_T1(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                     setup_awg=setup_awg, **kw)

        if qubsrc_power is not None:
            station.qubsrc.power(qubsrc_power)
        if hetsrc_power is not None:
            station.hetsrc.RF.power(hetsrc_power)

        if qubsrc_freq == 'dict':
            station.qubsrc.frequency(d["fq"])
        elif qubsrc_freq == None:
            pass
        else:
            station.qubsrc.frequency(qubsrc_freq)

        if hetsrc_freq == 'dict':
            station.hetsrc.frequency(d["f0"])
        elif hetsrc_freq == None:
            pass
        else:
            station.hetsrc.frequency(hetsrc_freq)

        station.qubsrc.modulation_rf('ON')
        station.qubsrc.output_rf('ON')
        station.RF.on()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('ON')
        station.RF.ref_LO_out('LO')
        station.LO.on()
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()

        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.qubsrc.modulation_rf('OFF')
        station.qubsrc.output_rf('OFF')
        station.RF.off()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('OFF')
        station.RF.ref_LO_out('OFF')
        station.LO.off()
        time.sleep(0.1)

        # it is unclear which combination of off and stop and sleep is required
        # but without them the timing goes wrong

        if fit:
            # rotate IQ data
            angle = dp.IQangle(mag*np.exp(1.j*phase))
            rotated_data = dp.IQrotate(mag*np.exp(1.j*phase), angle)

            # fit T1; still needs testing for robustness
            mod = lmfit.models.ExpressionModel('off + amp * exp(-x/t1)')
            xdat = times
            ydat = np.real(rotated_data)
            if T1_guess == None:
                T1_estimate = 1.5e-6
            elif T1_guess == 'dict':
                if "t1" in list(d.keys()) and d["t1"] < 40e-6:
                    T1_estimate = d["t1"]
                else:
                    T1_estimate = 1.5e-6
            else:
                T1_estimate = T1_guess
            params = mod.make_params(
                off=ydat[-1], amp=ydat[0]-ydat[-1], t1=T1_estimate)
            params['t1'].set(min=1e-9)
            params['t1'].set(max=40e-6)
            out = mod.fit(ydat, params, x=xdat)
            T1_time = out.params['t1']
            d["t1"] = T1_time

            return [times, mag, phase, T1_time]

        else:
            return [times, mag, phase]

    if fit == True:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="T1" + str(suffix),
                          unit="s",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
        ])


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
        seq.setup_awg(delays=delays, pulse_time=pulse_time,
                      readout_time=readout_time, cycle_time=20e-6, start_awg=True)

    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(True)
    controller.setup_acquisition(
        samples=None, records=delays.size, buffers=navgs, acq_time=acq_time, verbose=False)


def measure_echo(controller, delays, pulse_time, readout_time, qubsrc_power=None, qubsrc_freq=None, hetsrc_power=None, hetsrc_freq=None,
                 navgs=500, acq_time=2.56e-6, setup_awg=True, suffix='', fit=False, **kw):

    def return_alazar_trace(d):
        station = d["STATION"]
        times = delays
        if pulse_time == 'dict':
            setup_echo(controller, delays, d['pipulse']/2, readout_time, navgs=navgs, acq_time=acq_time,
                       setup_awg=True, **kw)
        else:
            setup_echo(controller, delays, pulse_time, readout_time, navgs=navgs, acq_time=acq_time,
                       setup_awg=setup_awg, **kw)

        if qubsrc_power is not None:
            station.qubsrc.power(qubsrc_power)
        if hetsrc_power is not None:
            station.hetsrc.RF.power(hetsrc_power)

        if qubsrc_freq == 'dict':
            station.qubsrc.frequency(d["fq"])
        elif qubsrc_freq == None:
            pass
        else:
            station.qubsrc.frequency(qubsrc_freq)

        if hetsrc_freq == 'dict':
            station.hetsrc.frequency(d["f0"])
        elif hetsrc_freq == None:
            pass
        else:
            station.hetsrc.frequency(hetsrc_freq)

        station.qubsrc.modulation_rf('ON')
        station.qubsrc.output_rf('ON')
        station.RF.on()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('ON')
        station.RF.ref_LO_out('LO')
        station.LO.on()
        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.awg.start()

        data = np.squeeze(controller.acquisition())[..., 0]
        mag, phase = np.abs(data), np.angle(data, deg=False)

        station.fg.ch1.state('OFF')
        station.awg.stop()
        station.qubsrc.modulation_rf('OFF')
        station.qubsrc.output_rf('OFF')
        station.RF.off()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('OFF')
        station.RF.ref_LO_out('OFF')
        station.LO.off()
        time.sleep(0.1)

        # it is unclear which combination of off and stop and sleep is required
        # but without them the timing goes wrong

        if fit:
            # rotate IQ data
            angle = dp.IQangle(mag*np.exp(1.j*phase))
            rotated_data = dp.IQrotate(mag*np.exp(1.j*phase), angle)

            # fit T2; still needs testing for robustness
            mod = lmfit.models.ExpressionModel('off + amp * exp(-x/t1)')
            xdat = times
            ydat = np.real(rotated_data)
            params = mod.make_params(
                off=ydat[-1], amp=ydat[0]-ydat[-1], t1=1e-6)
            params['t1'].set(min=1e-9)
            out = mod.fit(ydat, params, x=xdat)
            T2_time = out.params['t1']

            return [times, mag, phase, T2_time]

        else:
            return [times, mag, phase]

    if fit == True:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="T2echo" + str(suffix),
                          unit="s",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(return_alazar_trace, [
            DataParameter(name="delay_time" + str(suffix),
                          unit="s",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["delay_time" + str(suffix)],
                          ),
        ])


def setup_QPP(controller, acq_time, navg, SR=250e6, setup_awg=True):
    """
    Set up ...
    """

    station = qcodes.Station.default

    # setting up the AWG
    if setup_awg:
        seq = QPTriggerSequence(station.awg, SR=1e7)
        seq.load_sequence(cycle_time=acq_time+1e-3, plot=False,
                          use_event_seq=True, ncycles=navg)

    controller.verbose = True
    controller.average_buffers(False)
    controller.average_buffers_postdemod(False)
    station.alazar.sample_rate(int(SR))
    npoints = int(acq_time*SR // 128 * 128)
    controller.setup_acquisition(npoints, 1, navg)
    print(controller, navg, acq_time, controller.demod_frq(), npoints)


def measure_QPP(controller, acq_time, navg, suffix='', SR=250e6, setup_awg=True, hetsrc_power=None, hetsrc_freq=None, **kw):

    @MakeMeasurementFunction(
        [
            DataParameter("timestamp" + str(suffix), "s", "array", False),
        ]
    )
    def return_alazar_trace(d):

        setup_QPP(controller, acq_time, navg, SR=SR, setup_awg=setup_awg, **kw)

        station = d["STATION"]

        station = d["STATION"]

        station.awg.start()

        if hetsrc_power is not None:
            station.hetsrc.RF.power(hetsrc_power)

        if hetsrc_freq == 'dict':
            station.hetsrc.frequency(d["f0"])
        elif hetsrc_freq == None:
            pass
        else:
            station.hetsrc.frequency(hetsrc_freq)

        station.qubsrc.modulation_rf('OFF')
        station.qubsrc.output_rf('OFF')
        station.RF.on()
        station.RF.pulsemod_source('EXT')
        station.RF.pulsemod_state('OFF')
        station.RF.ref_LO_out('LO')
        station.LO.on()

        station.alazar.clear_buffers()
        data = np.squeeze(controller.acquisition())[..., 0]
        time.sleep(0.1)

        station.awg.stop()
        station.qubsrc.output_rf('OFF')
        station.RF.off()
        station.RF.ref_LO_out('OFF')
        station.LO.off()

        timestamp = int(time.time()*1e6)
        datasaver_run_id = d["DATASET"].run_id
        data_folder_path = str(qcodes.config.core.db_location)[:-3]+"\\"
        Path(data_folder_path).mkdir(parents=True, exist_ok=True)
        np.save(data_folder_path+"ID_" +
                f"{datasaver_run_id}_IQ_{timestamp:d}", [controller.demod_tvals, data])

        return [timestamp]
    return return_alazar_trace


def setup_single_averaged_IQpoint(controller, time_bin, integration_time, setup_awg=True,
                                  post_integration_delay=10e-6,
                                  verbose=True, allocated_buffers=None):
    """
    Under development! If you end up wanting to use this, be aware that it is buggy/that it still needs to be finished.
    Arno can help.
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
    """
    Under development! If you end up wanting to use this, be aware that it is buggy/that it still needs to be finished.
    Arno can help.
    """

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
    """
    Under development! If you end up wanting to use this, be aware that it is buggy/that it still needs to be finished.
    Arno can help.
    """
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
    """
    Under development! If you end up wanting to use this, be aware that it is buggy/that it still needs to be finished.
    Arno can help.
    """
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
