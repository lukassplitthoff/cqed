# Authors: Lukas Splitthoff, Lukas Gruenhaupt, Arno Bargerbos @TUDelft
# 04-DEC-2019
# some strategies I had in mind for the resonance finder: increase span, increase navgs, increase npts, adjust filtering parameters, take previous value
"""
A set of functions to conveniently use for VNA measurements with pysweep.
"""

from pysweep.core.measurementfunctions import MakeMeasurementFunction
from pysweep.databackends.base import DataParameter
import numpy as np


def setup_frq_sweep(
    station,
    fstart,
    fstop,
    fstep,
    chan="S21",
    bw=None,
    navgs=None,
    pwr=None,
    electrical_delay=None,
):
    """Function that sets up the VNA according to the specified parameters, leaving
    other parameters intact. Frequency parameters are required, others are optional.
    Assumes that a channel with name `chan` is already created.

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        fstart (Hz): starting frequency of VNA sweep
        fstop (Hz): final frequency of VNA sweep
        fstep (Hz): step size of VNA sweep
        chan: name of VNA channel to be used
        bw (Hz): VNA bandwidth
        navgs: number of averages per measurement
        navgs: number of averages per measurement
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA

    """

    vna_trace = getattr(station.vna.channels, chan)
    try: 
        vna_trace.setup_lin_sweep() #gets you out of CW in Jaap's new ZNB class
    except:
        pass
        
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


@MakeMeasurementFunction(
    [
        DataParameter("frequency", "Hz", "array", True),
        DataParameter("amplitude", "", "array"),
        DataParameter("phase", "rad", "array"),
    ]
)
def return_vna_trace(d):
    """Pysweep VNA measurement function that returns an S21 trace given the currently
    set VNA parameters.

    Returns:
        VNA frequency axis in Hz, linear magnitude? in V?, phase in radians.

    """
    station = d["STATION"]
    freqs = np.linspace(
        station.vna.S21.start(), station.vna.S21.stop(), station.vna.S21.npts()
    )

    if not station.vna.rf_power():
        station.vna.rf_on()

    vna_data = station.vna.S21.trace_mag_phase()

    return [freqs, vna_data[0], vna_data[1]]

def transmission_vs_frequency_explicit(center, span, suffix=None):
    """Pysweep VNA measurement function that returns an S21 trace given the currently
    set VNA parameters for a given center frequency a given span and a given suffix.
    This can be useful when one wants to measure the response of several resonances
    versus a single parameter.

    Args:
        center (Hz): center frequency of VNA trace
        suffix (int): index of measurement function


    Returns:
    Pysweep measurement function

    """

    @MakeMeasurementFunction(
        [
            DataParameter(
                name="frequency" + str(suffix),
                unit="Hz",
                paramtype="array",
                # yes independent, but pysweep will not recognize it as such
                independent=2,
            ),
            DataParameter(
                name="amplitude" + str(suffix),
                unit="",
                paramtype="array",
                # explicitely tell that this parameter depends on
                # the corresponding frequency parameter
                extra_dependencies=["frequency" + str(suffix)],
            ),
            DataParameter(
                name="phase" + str(suffix),
                unit="rad",
                paramtype="array",
                # explicitely tell that this parameter depends on
                # the corresponding frequency parameter
                extra_dependencies=["frequency" + str(suffix)],
            ),
        ]
    )
    def transmission_vs_frequency_measurement_function(d):
        station = d["STATION"]
        station.vna.S21.center(center)
        station.vna.S21.span(span)
        return return_vna_trace(d)

    # return a measurement function
    return transmission_vs_frequency_measurement_function


def multiple_meas_functions(freq_list, span_list):
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
    for i, c in enumerate(freq_list):
        s = span_list[i]
        fun_str += "cvna.transmission_vs_frequency_explicit({}, {}, suffix={})+".format(
            c, s, i
        )
    fun_str = fun_str[:-1]
    return fun_str

def transmission_vs_frequency_explicit_extended(fstart, fstop, fstep, bw=None, navgs=None, pwr=None, electrical_delay=None, suffix=None, chan="S21"):
    """Pysweep VNA measurement function that returns an S21 trace given the currently
    set VNA parameters for a given center frequency a given span and a given suffix.
    This can be useful when one wants to measure the response of several resonances
    versus a single parameter.

    Args:
        center (Hz): center frequency of VNA trace
        suffix (int): index of measurement function


    Returns:
    Pysweep measurement function

    """

    @MakeMeasurementFunction(
        [
            DataParameter(
                name="frequency" + str(suffix),
                unit="Hz",
                paramtype="array",
                # yes independent, but pysweep will not recognize it as such
                independent=2,
            ),
            DataParameter(
                name="amplitude" + str(suffix),
                unit="",
                paramtype="array",
                # explicitely tell that this parameter depends on
                # the corresponding frequency parameter
                extra_dependencies=["frequency" + str(suffix)],
            ),
            DataParameter(
                name="phase" + str(suffix),
                unit="rad",
                paramtype="array",
                # explicitely tell that this parameter depends on
                # the corresponding frequency parameter
                extra_dependencies=["frequency" + str(suffix)],
            ),
        ]
    )
    def transmission_vs_frequency_measurement_function(d):
        station = d["STATION"]
        vna_trace = getattr(station.vna.channels, chan)
        try: 
            vna_trace.setup_lin_sweep() #gets you out of CW in Jaap's new ZNB class
        except:
            pass

        vna_trace.start(int(fstart))
        vna_trace.stop(int(fstop))
        vna_trace.npts(int((fstop - fstart) / fstep))
        if navgs is not None:
            vna_trace.avg(navgs)
        if bw is not None:
            vna_trace.bandwidth(bw)
        if pwr is not None:
            vna_trace.power(pwr)
        if electrical_delay is not None:
            vna_trace.electrical_delay(electrical_delay)

        return return_vna_trace(d)

    # return a measurement function
    return transmission_vs_frequency_measurement_function

def multiple_meas_functions_extended(fstart_list, fstop_list, fstep_list, bw_list=None, navgs_list=None, pwr_list=None, electrical_delay_list=None,):
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
    for ii, fstart in enumerate(fstart_list):
        fstop = fstop_list[ii]
        fstep = fstep_list[ii]
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

        fun_str += "cvna.transmission_vs_frequency_explicit_extended(fstart={}, fstop={}, fstep={}, bw={}, navgs={}, pwr={}, electrical_delay={}, suffix={})+".format(
            fstart, fstop, fstep, bw, navgs, pwr, delay, ii
        )
    fun_str = fun_str[:-1]
    return fun_str


def measure_resonance_estimate(f0, fspan, fstep, res_finder, **kwargs):
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
        setup_frq_sweep(
            station=station,
            fstart=d["f0"] - fspan / 2,
            fstop=d["f0"] + fspan / 2,
            fstep=fstep,
            **kwargs
        )
        # try:
        #     station.qubsrc.off()
        # except:
        #     pass
        cal = return_vna_trace(d)
        m0 = res_finder(cal[0], 20 * np.log10(cal[1]))
        if m0 == None:
            raise Exception(
                "Failed to find a resonance."
            )  # needs work, can implement alternative strategies
        d["f0"] = m0
        return [m0]

    return resonance_estimate_measurement_function


def measure_S21_adaptive(f0, fspan, fstep, **kwargs):
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
            DataParameter("frequency", "Hz", "array", True),
            DataParameter("amplitude", "", "array"),
            DataParameter("phase", "rad", "array"),
        ]
    )
    def adaptive_measurement_function(d):
        station = d["STATION"]

        if "f0" not in d:
            d["f0"] = f0
        setup_frq_sweep(
            station=station,
            fstart=d["f0"] - fspan / 2,
            fstop=d["f0"] + fspan / 2,
            fstep=fstep,
            **kwargs
        )
        data = return_vna_trace(d)
        return [data[0], data[1], data[2]]

    return adaptive_measurement_function

def setup_frq_sweep_CW(
    station,
    cw_frequency,
    npts,
    chan="S21",
    bw=None,
    pwr=None,
    electrical_delay=None,
):
    """Function that sets up the VNA according to the specified parameters, leaving
    other parameters intact. Frequency parameters are required, others are optional.
    Assumes that a channel with name `chan` is already created. To use in CW mode.

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        fstart (Hz): starting frequency of VNA sweep
        fstop (Hz): final frequency of VNA sweep
        fstep (Hz): step size of VNA sweep
        chan: name of VNA channel to be used
        bw (Hz): VNA bandwidth
        navgs: number of averages per measurement
        navgs: number of averages per measurement
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA

    """

    vna_trace = getattr(station.vna.channels, chan)
    try: 
        vna_trace.setup_cw_sweep() #gets you into CW in Jaap's new ZNB class
    except:
        print("CW Mode does not exist in this qcodes version")

    if bw is None:
        bw = vna_trace.bandwidth()
    if pwr is None:
        pwr = vna_trace.power()
    if electrical_delay is None:
        electrical_delay = vna_trace.electrical_delay()

    vna_trace.cw_frequency(int(cw_frequency))
    vna_trace.npts(npts)
    vna_trace.bandwidth(bw)
    vna_trace.power(pwr)
    vna_trace.electrical_delay()

@MakeMeasurementFunction(
    [
        DataParameter("time", "s", "array", True),
        DataParameter("I", "", "array"),
        DataParameter("Q", "", "array"),
    ]
)
def return_vna_trace_CW(d):
    """Pysweep VNA measurement function that returns ..

    Returns:
        VNA ..

    """
    station = d["STATION"]
    bw = station.vna.S21.bandwidth()
    sweep_time = station.vna.S21.sweep_time()
    npts = station.vna.S21.npts()
    times = np.linspace(1/bw, sweep_time, npts)

    if not station.vna.rf_power():
        station.vna.rf_on()

    vna_data = station.vna.S21.trace_fixed_frequency()

    return [times, vna_data[0], vna_data[1]]
    
def measure_SNR_two_frequencies(f0, f1):
    @MakeMeasurementFunction([DataParameter(name='signal_0', unit='dB'), 
                            DataParameter(name='signal_1', unit='dB'), 
                            DataParameter(name='SNR', unit='dB')])
    def return_SNR_CW(d):
        """Pysweep VNA measurement function that returns ..

        Returns:
            VNA ..

        """
        station = d["STATION"]
        station.vna.S21.cw_frequency(f0)

        if not station.vna.rf_power():
            station.vna.rf_on()

        vna_data = station.vna.S21.trace_fixed_frequency()
        I = vna_data[0]
        Q = vna_data[1]
        S = I + 1j*Q
        lin_mean_f0 = np.abs(S.mean())

        station.vna.S21.cw_frequency(f1)

        vna_data = station.vna.S21.trace_fixed_frequency()
        I = vna_data[0]
        Q = vna_data[1]
        S = I + 1j*Q
        lin_mean_f1 = np.abs(S.mean())
        
        return [20*np.log10(lin_mean_f0), 20*np.log10(lin_mean_f1), 20*np.log10(lin_mean_f0/lin_mean_f1)]
    return return_SNR_CW


@MakeMeasurementFunction([DataParameter(name='signal', unit='dB'), 
                          DataParameter(name='noise', unit='dB'), 
                          DataParameter(name='SNR', unit='dB')])
def return_SNR_CW(d):
    """Pysweep VNA measurement function that returns ..

    Returns:
        VNA ..

    """
    station = d["STATION"]

    if not station.vna.rf_power():
        station.vna.rf_on()

    vna_data = station.vna.S21.trace_fixed_frequency()
    I = vna_data[0]
    Q = vna_data[1]
    S = I + 1j*Q
    lin_mean = np.abs(S.mean())
    lin_std = np.abs(S.std())
    
    return [20*np.log10(lin_mean), 20*np.log10(lin_std), 20*np.log10(lin_mean/lin_std)]


def measure_PSD_averaged(f0, bandwidth, points, averages, **kwargs):
    @MakeMeasurementFunction(
        [
            DataParameter("Frequency", "Hz", "array", True),
            DataParameter("PSDI", "pwr", "array"),
            DataParameter("PSDQ", "pwr", "array"),
        ]
    )
    def return_PSD_CW(d):
        """Pysweep VNA measurement function that returns .... Work in progress!

        Returns:
            VNA ..

        """
        station = d["STATION"]

        if not station.vna.rf_power():
            station.vna.rf_on()
        setup_frq_sweep_CW(station=station, cw_frequency=f0, npts=points, bw=bandwidth, **kwargs)

        for ii in range(averages):
            vna_data = station.vna.S21.trace_fixed_frequency()
            I = vna_data[0]
            Q = vna_data[1]

            bw = station.vna.S21.bandwidth()
            sweep_time = station.vna.S21.sweep_time()
            npts = station.vna.S21.npts()
            times = np.linspace(1/bw, sweep_time, npts)

            n = times.size
            step = np.diff(times)[0]
            fftfreq = np.fft.fftfreq(n, d=step)
            fft_Q = np.abs(np.fft.fft(Q-np.mean(Q)))**2
            fft_I = np.abs(np.fft.fft(I-np.mean(I)))**2
            if ii == 0:
                fft_Q_array = fft_Q
                fft_I_array = fft_I
            else:
                fft_Q_array = (fft_Q_array*(ii)+fft_Q)/(ii+1)
                fft_I_array = (fft_I_array*(ii)+fft_I)/(ii+1)

        return [fftfreq, fft_Q_array, fft_I_array]
                
    return return_PSD_CW


@MakeMeasurementFunction([DataParameter(name='amplitude', unit=''), 
                          DataParameter(name='phase', unit='rad')])
def return_vna_point_CW(d):
    station = d["STATION"]
    return list(station.vna.S21.point_fixed_frequency_mag_phase())


def measure_CW_optimized(cw_frequency, suffix='', qubsrc_power=None, ignore_dict=False, **kwargs):
    """Pysweep VNA measurement function that measures in CW at frequency
    cw_frequency.

    cw_frequency can be updated through other measurement or sweep functions (such as `measure_resonance_estimate`)
    This is helpful when measuring versus parameters that change the resonance frequency

    Args:
    cw_frequency (Hz): Frequency at which to measure. Ignored if there is already an f0 in d['f0'].
    qubsrc_power (dBm): power of a qubsrc in the station. Hardcoded, should be more subtle.
    ignore_dict (boolean): if true, cw_frequency is taken only from the input, not from the dictionary. 
    Useful when doing multiple CW measurements at different frequencies in the same measurement. But not very clever.
    kwargs: see `setup_frq_sweep_CW`.

    Returns:
    Pysweep measurement function

    """

    @MakeMeasurementFunction(
        [
            DataParameter("amplitude"+suffix, "", "array"),
            DataParameter("phase"+suffix, "rad", "array"),
        ]
    )
    def cw_measurement_function(d):
        station = d["STATION"]
        if qubsrc_power != None:
            station.qubsrc.power(qubsrc_power)
            station.qubsrc.output_rf('ON')
        if ignore_dict == False:
            if "f0" not in d:
                d["f0"] = cw_frequency
            setup_frq_sweep_CW(
                station=station,
                cw_frequency=d["f0"],
                **kwargs
            )
        else:
            setup_frq_sweep_CW(
                station=station,
                cw_frequency=cw_frequency,
                **kwargs
            )
        data = return_vna_point_CW(d)
        if qubsrc_power != None:
            station.qubsrc.output_rf('OFF') #its good for it to be off when you measure the resonator

        return [data[0], data[1]]

    return cw_measurement_function


