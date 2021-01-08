# Authors: Lukas Splitthoff, Lukas Gruenhaupt, Arno Bargerbos @TUDelft
"""
A set of functions to conveniently use for VNA measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction, MeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np
import time

# linear VNA sweeps


def setup_linear_sweep(
    station,
    fstart=None,
    fstop=None,
    fstep=None,
    npts=None,
    center=None,
    span=None,
    chan="S21",
    bw=None,
    navgs=None,
    pwr=None,
    electrical_delay=None,
):
    """Function that sets up the VNA in linear sweep mode according to the specified parameters, leaving
    other parameters intact. All parameters are optional, but one has to choose between using span, center, or fstop, fstart.
    One also has to choose between npts or fstep. Specifying both leads to unintended behaviour (warnings not yet implemented).
    Assumes that a channel with name `chan` is already created.

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        fstart (Hz): starting frequency of VNA sweep
        fstop (Hz): final frequency of VNA sweep
        fstep (Hz): step size of VNA sweep
        npts (int): number of steps between fstart and fstop
        center (Hz): center frequency of the trace of width span
        span (Hz): span of the trace centered at center
        chan: name of VNA channel to be used
        bw (Hz): VNA bandwidth
        navgs: number of averages per measurement
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA

    """

    vna_trace = getattr(station.vna.channels, chan)
    try:
        # gets you out of CW in Jaap's new ZNB class, can remove try when it is in main qcodes
        vna_trace.setup_lin_sweep()
    except:
        pass

    if span is not None and center is not None:
        fstart = center - span / 2
        fstop = center + span / 2
    if fstart is None:
        fstart = station.vna.S21.start()
    if fstop is None:
        fstop = station.vna.S21.stop()
    if npts is None and fstep is not None:
        npts = int((fstop - fstart) / fstep)
    if navgs is None:
        navgs = vna_trace.avg()
    if bw is None:
        bw = vna_trace.bandwidth()
    if pwr is None:
        pwr = vna_trace.power()
    if electrical_delay is None:
        electrical_delay = vna_trace.electrical_delay()

    vna_trace.start(int(fstart))
    vna_trace.stop(int(fstop))
    vna_trace.npts(int((fstop - fstart) / fstep))
    vna_trace.bandwidth(bw)
    vna_trace.power(pwr)
    vna_trace.avg(navgs)
    vna_trace.electrical_delay()


def return_linear_sweep(suffix='', setup_vna=False, **kwargs):
    """Pysweep VNA measurement function that returns an S21 trace, either given the currently
    set VNA parameters or for a custom set when setup_vna=true.
    By setting suffix one can measure the response of several resonances
    versus a parameter simultaneously.

    Args:
        suffix (int): index of measurement function
        setup_vna (boolean): whether to use the current VNA settings or pass new ones via kwargs.
        kwargs: see `setup_linear_sweep`.


    Returns:
    Pysweep measurement function
    """

    @MakeMeasurementFunction(
        [
            DataParameter(
                name="frequency" + str(suffix),
                unit="Hz",
                paramtype="array",
                independent=2,
            ),
            DataParameter(
                name="amplitude" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
            DataParameter(
                name="phase" + str(suffix),
                unit="rad",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
        ]
    )
    def transmission_vs_frequency_measurement_function(d):
        station = d["STATION"]
        if setup_vna:
            setup_linear_sweep(station=station, **kwargs)

        freqs = np.linspace(station.vna.S21.start(),
                            station.vna.S21.stop(), station.vna.S21.npts())

        if not station.vna.rf_power():
            station.vna.rf_on()

        vna_data = station.vna.S21.trace_mag_phase()
        station.vna.rf_off()

        return [freqs, vna_data[0], vna_data[1]]

    return transmission_vs_frequency_measurement_function


def measure_multiple_linear_sweeps(
    fstart_list=None,
    fstop_list=None,
    fstep_list=None,
    npts_list=None,
    center_list=None,
    span_list=None,
    bw_list=None,
    navgs_list=None,
    pwr_list=None,
    electrical_delay_list=None,
):
    """Combines several measurement functions into one, allowing for measuring N
    resonances using a single measurement function.

    Should be renamed or made more general.
    Args:
        freq_list (Hz): list containing center frequencies
        span_list (Hz): list containing spans


    Returns:
    Pysweep measurement function

    """
    fun_str = ""

    if fstart_list is not None:
        main_list = fstart_list
    elif center_list is not None:
        main_list = center_list

    for ii in range(len(main_list)):
        if fstart_list is not None:
            fstart = fstart_list[ii]
        else:
            fstart = None
        if fstop_list is not None:
            fstop = fstop_list[ii]
        else:
            fstop = None
        if fstep_list is not None:
            fstep = fstep_list[ii]
        else:
            fstep = None
        if npts_list is not None:
            npts = npts_list[ii]
        else:
            npts = None
        if center_list is not None:
            center = center_list[ii]
        else:
            center = None
        if span_list is not None:
            span = span_list[ii]
        else:
            span = None
        if bw_list is not None:
            bw = bw_list[ii]
        else:
            bw = None
        if navgs_list is not None:
            navgs = navgs_list[ii]
        else:
            navgs = None
        if pwr_list is not None:
            pwr = pwr_list[ii]
        else:
            pwr = None
        if electrical_delay_list is not None:
            delay = electrical_delay_list[ii]
        else:
            delay = None

        fun_str += "cvna.transmission_vs_frequency(suffix={}, setup_vna=True, fstart={}, fstop={}, fstep={}, npts={}, center={}, span={}, bw={}, navgs={}, pwr={}, electrical_delay={})+".format(
            fstart, fstop, fstep, npts, center, span, bw, navgs, pwr, delay, ii
        )
    fun_str = fun_str[:-1]
    return fun_str


def measure_resonance_frequency(f0, span, fstep, res_finder, save_trace=False, suffix='', **kwargs):
    """Pysweep VNA measurement function that measures an estimated resonance
    frequency. Can choose to save the trace or only the resonance frequency.

    f0 (Hz): Frequency around which to measure. Ignored if there is already an f0 in d['f0'].
    fspan (Hz): Span around f0 to measure.
    fstep (Hz): Frequency step size.
    res finder: Function that finds a resonance from VNA output. WIP.
    kwargs: see `setup_linear_sweep`.

    Returns:
    Pysweep measurement function

    Returns:
    Pysweep measurement function

    """
    def resonance_estimate_measurement(d):
        if "f0" not in d:
            d["f0"] = f0
        freqs, mag, phase = return_linear_sweep(
            suffix=suffix, setup_vna=True, span=span, center=f0, fstep=fstep, **kwargs)(d)
        m0 = res_finder(freqs, mag)
        if m0 == None:
            raise Exception(
                "Failed to find a resonance."
            )  # needs work, can implement alternative strategies
        d["f0"] = m0
        if save_trace == True:
            return [freqs, mag, phase, m0]
        else:
            return [m0]

    if save_trace == True:
        return MeasurementFunction(resonance_estimate_measurement, [
            DataParameter(name="frequency" + str(suffix),
                          unit="Hz",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["frequency" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["frequency" + str(suffix)],
                          ),
            DataParameter(name="resonance_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(resonance_estimate_measurement, [
            DataParameter(name="resonance_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])


def measure_adaptive_linear_sweep(f0, fspan, fstep, suffix='', **kwargs):
    """Pysweep VNA measurement function that measures S21 in a window around a
    frequency f0.

    f0 can be updated through other measurement or sweep functions (such as `measure_resonance_estimate`)
    This is helpful when measuring S21 versus parameters that change the resonance frequency

    Args:
    f0 (Hz): Frequency around which to measure. Ignored if there is already an f0 in d['f0'].
    fspan (Hz): Span around f0 to measure.
    fstep (Hz): Frequency step size.
    kwargs: see `setup_frq_sweep`.

    Returns:
    Pysweep measurement function

    """

    @MakeMeasurementFunction(
        [
            DataParameter(
                name="frequency" + str(suffix),
                unit="Hz",
                paramtype="array",
                independent=2,
            ),
            DataParameter(
                name="amplitude" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
            DataParameter(
                name="phase" + str(suffix),
                unit="rad",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
        ]
    )
    def adaptive_measurement_function(d):
        if "f0" not in d:
            d["f0"] = f0
        data = return_linear_sweep(
            suffix=suffix, setup_vna=True, span=fspan, center=f0, fstep=fstep, **kwargs)(d)
        return [data[0], data[1], data[2]]

    return adaptive_measurement_function

# CW functions from here onwards


def setup_CW_sweep(
    station, cw_frequency=None, t_int=None, npts=None, chan="S21", bw=None, pwr=None, electrical_delay=None,
):
    """Function that sets up the VNA according to the specified parameters, leaving
    other parameters intact. Frequency parameters are required, others are optional.
    Assumes that a channel with name `chan` is already created. To use in CW mode.

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        cw_frequency (Hz): frequency at which the CW sweep is performed
        npts (int): number of points in the CW sweep. Between 1 and 1e5 (or 1e6?)
        t_int (s): total time the measurement takes (npts/bw)
        chan: name of VNA channel to be used
        bw (Hz): VNA bandwidth
        navgs: number of averages per measurement
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA

    """

    vna_trace = getattr(station.vna.channels, chan)
    try:
        vna_trace.setup_cw_sweep()  # gets you into CW in Jaap's new ZNB class
    except:
        print("CW Mode does not exist in this qcodes version")

    if cw_frequency is None:
        cw_frequency = vna_trace.cw_frequency()
    if bw is None:
        bw = vna_trace.bandwidth()
    if npts is None and t_int is not None:
        npts = int(np.round(t_int*bw))
    elif npts is None:
        npts = vna_trace.npts()
    if pwr is None:
        pwr = vna_trace.power()
    if electrical_delay is None:
        electrical_delay = vna_trace.electrical_delay()

    vna_trace.cw_frequency(int(cw_frequency))
    vna_trace.npts(npts)
    vna_trace.bandwidth(bw)
    vna_trace.power(pwr)
    vna_trace.electrical_delay()


def return_cw_sweep(suffix='', setup_vna=False, **kwargs):
    @MakeMeasurementFunction(
        [
            DataParameter(
                name="time" + str(suffix),
                unit="s",
                paramtype="array",
                independent=2,
            ),
            DataParameter(
                name="I" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["time" + str(suffix)],
            ),
            DataParameter(
                name="Q" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["time" + str(suffix)],
            ),
        ]
    )
    def measure_cw_sweep(d):
        """Pysweep VNA measurement function that returns ..

        Returns:
            VNA ..

        """
        station = d["STATION"]
        if setup_vna:
            setup_CW_sweep(station=station, **kwargs)

        bw = station.vna.S21.bandwidth()
        sweep_time = station.vna.S21.sweep_time()
        npts = station.vna.S21.npts()
        times = np.linspace(1 / bw, sweep_time, npts)

        if not station.vna.rf_power():
            station.vna.rf_on()

        vna_data = station.vna.S21.trace_fixed_frequency()
        station.vna.rf_off()

        return [times, vna_data[0], vna_data[1]]

    return measure_cw_sweep


def measure_PSD_averaged(f0, bandwidth, averages, t_int=None, points=None, suffix='', **kwargs):
    """Pysweep VNA measurement function that returns ..

    Returns:
        VNA ..

    """
    @MakeMeasurementFunction(
        [
            DataParameter(
                name="frequency" + str(suffix),
                unit="s",
                paramtype="array",
                independent=2,
            ),
            DataParameter(
                name="PSD_I" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
            DataParameter(
                name="PSD_Q" + str(suffix),
                unit="",
                paramtype="array",
                extra_dependencies=["frequency" + str(suffix)],
            ),
        ]
    )
    def return_PSD_CW(d):
        """Pysweep VNA measurement function that returns .... Work in progress!

        Returns:
            VNA ..

        """
        station = d["STATION"]

        for ii in range(averages):
            vna_data = return_cw_sweep(
                setup_vna=True, cw_frequency=f0, bw=bandwidth, t_int=t_int, npoints=points, **kwargs)(d)
            I = vna_data[1]
            Q = vna_data[2]

            bw = station.vna.S21.bandwidth()
            sweep_time = station.vna.S21.sweep_time()
            npts = station.vna.S21.npts()
            times = np.linspace(1 / bw, sweep_time, npts)

            n = times.size
            step = np.diff(times)[0]
            fftfreq = np.fft.fftfreq(n, d=step)
            fft_Q = np.abs(np.fft.fft(Q - np.mean(Q))) ** 2
            fft_I = np.abs(np.fft.fft(I - np.mean(I))) ** 2
            if ii == 0:
                fft_Q_array = fft_Q
                fft_I_array = fft_I
            else:
                fft_Q_array = (fft_Q_array * (ii) + fft_Q) / (ii + 1)
                fft_I_array = (fft_I_array * (ii) + fft_I) / (ii + 1)

        return [fftfreq, fft_Q_array, fft_I_array]

    return return_PSD_CW


def measure_SNR_CW(**kwargs):
    @MakeMeasurementFunction(
        [
            DataParameter(name="signal", unit="dB"),
            DataParameter(name="noise", unit="dB"),
            DataParameter(name="SNR", unit="dB"),
        ]
    )
    def return_SNR_CW(d):
        """Pysweep VNA measurement function that returns ..

        Returns:
            VNA ..

        """
        station = d["STATION"]

        vna_data = return_cw_sweep(setup_vna=True, **kwargs)(d)
        I = vna_data[1]
        Q = vna_data[2]
        S = I + 1j * Q
        lin_mean = np.abs(S.mean())
        lin_std = np.abs(S.std())

        return [
            20 * np.log10(lin_mean),
            20 * np.log10(lin_std),
            20 * np.log10(lin_mean / lin_std),
        ]
    return return_SNR_CW


def return_cw_point(suffix='', setup_vna=False, **kwargs):
    @MakeMeasurementFunction(
        [DataParameter(name="amplitude" + str(suffix), unit=""),
         DataParameter(name="phase" + str(suffix), unit="rad")]
    )
    def return_vna_point_CW(d):
        station = d["STATION"]
        if setup_vna:
            setup_CW_sweep(station=station, **kwargs)

        if not station.vna.rf_power():
            station.vna.rf_on()

        data = list(station.vna.S21.point_fixed_frequency_mag_phase())

        station.vna.rf_off()

        return data
    return return_vna_point_CW

def measure_2tone_sweep(frequencies, cw_frequency=None, qubsrc_power=None, suffix='', settling_time=10e-6, **kwargs):

    def cw_measurement_function(d):
        station = d["STATION"]
        station.qubsrc.output_rf('ON')
        station.qubsrc.modulation_rf('OFF')


        if qubsrc_power != None:
            station.qubsrc.power(qubsrc_power)

        mag = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)

        if cw_frequency == 'dict':
            setup_CW_sweep(station=station, cw_frequency=d["f0"], **kwargs)
        else:
            setup_CW_sweep(station=station, cw_frequency=cw_frequency, **kwargs)

        for ii in range(len(frequencies)):
            station.qubsrc.frequency(frequencies[ii])
            time.sleep(settling_time)
            data = return_cw_point()(d)
            mag[ii] = data[0]
            phase[ii] = data[1]

        station.qubsrc.output_rf('OFF')

        return [frequencies, mag, phase]

    return MeasurementFunction(cw_measurement_function, [
        DataParameter(name="frequency" + str(suffix),
                      unit="Hz",
                      paramtype="array",
                      independent=2,
                      ),
        DataParameter(name="amplitude" + str(suffix),
                      unit="",
                      paramtype="array",
                      extra_dependencies=["frequency" + str(suffix)],
                      ),
        DataParameter(name="phase" + str(suffix),
                      unit="rad",
                      paramtype="array",
                      extra_dependencies=["frequency" + str(suffix)],
                      ),
    ])


def measure_qubit_frequency(frequencies, cw_frequency=None, qubsrc_power=None, suffix='', settling_time=10e-6, save_trace=True, res_finder=None, **kwargs):

    def cw_measurement_function(d):
        station = d["STATION"]

        freqs, mag, phase = measure_2tone_sweep(frequencies=frequencies, cw_frequency=cw_frequency,
                                                qubsrc_power=qubsrc_power, suffix=suffix, settling_time=settling_time, **kwargs)(d)

        m0 = res_finder(freqs, mag)

        if m0 == None:
            raise Exception(
                "Failed to find a resonance."
            )  # needs work, can implement alternative strategies
        d["fq"] = m0
  
        if save_trace == True:
            return [frequencies, mag, phase, m0]
        else:
            return [m0]

    if save_trace == True:
        return MeasurementFunction(cw_measurement_function, [
            DataParameter(name="frequency" + str(suffix),
                          unit="Hz",
                          paramtype="array",
                          independent=2,
                          ),
            DataParameter(name="amplitude" + str(suffix),
                          unit="",
                          paramtype="array",
                          extra_dependencies=["frequency" + str(suffix)],
                          ),
            DataParameter(name="phase" + str(suffix),
                          unit="rad",
                          paramtype="array",
                          extra_dependencies=["frequency" + str(suffix)],
                          ),
            DataParameter(name="qubit_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])
    else:
        return MeasurementFunction(cw_measurement_function, [
            DataParameter(name="qubit_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])
