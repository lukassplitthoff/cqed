# Authors: Lukas Splitthoff, Lukas Gruenhaupt, Arno Bargerbos, Marta Pita Vidal @TUDelft
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
    bw=None,
    navgs=None,
    pwr=None,
    electrical_delay=None,
    chan="S21",
):
    """Function that sets up the VNA in linear sweep mode according to the specified parameters, leaving
    other parameters intact. All parameters are optional, but one has to choose between using span, center, or fstop, fstart.
    Otherwise fstop and fstart will get overwritten by span and center.
    One also has to choose between npts or fstep, otherwise npts is used (warnings not yet implemented).
    Assumes that a channel with name `chan` is already created.

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        fstart (Hz): starting frequency of VNA sweep
        fstop (Hz): final frequency of VNA sweep
        fstep (Hz): step size of VNA sweep
        npts (int): number of steps between fstart and fstop
        center (Hz): center frequency of the trace of width span
        span (Hz): span of the trace centered at center
        bw (Hz): VNA bandwidth
        navgs: number of averages per measurement
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA
        chan: name of VNA channel to be used

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
    set VNA parameters or for a custom set of parameters specified via kwargs when setup_vna=true.
    By setting suffix one can measure the response of several resonances
    versus a parameter in the same measurement without the parameter names interfering.

    Args:
        suffix (int): suffix added to the DataParameters.
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
    def measurement_function(d):
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

    return measurement_function


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
    """Helper function that combines several measurement functions into one, allowing for measuring N
    resonances using a single measurement function. 
    Convenient when N is large; otherwise you can just use + to concatenate several instances of `return_linear_sweep`.

    Args are the same as as `setup_linear_sweep` but now input as lists/arrays.

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

        fun_str += "cvna.return_linear_sweep(suffix={}, setup_vna=True, fstart={}, fstop={}, fstep={}, npts={}, center={}, span={}, bw={}, navgs={}, pwr={}, electrical_delay={})+".format(
            fstart, fstop, fstep, npts, center, span, bw, navgs, pwr, delay, ii
        )
    fun_str = fun_str[:-1]
    return fun_str


def measure_resonance_frequency(peak_finder, save_trace=False, suffix='', **kwargs):
    """Pysweep VNA measurement function that can estimate a resonance frequency `f0`. 
    frequency. Can choose to save the trace or only the resonance frequency.

    peak_finder: Function that finds a peak from VNA output. See for example general_tools -> peak_finding.py
    save_trace (boolean): whether to save the full VNA trace and the determined f0 or only f0.
    suffix (int): suffix added to the DataParameters.
    kwargs: see `setup_linear_sweep`.

    Returns:
    Pysweep measurement function

    """
    def measurement_function(d):
        if bool(kwargs):  # checks if there are kwargs, otherwise we can skip setting up the VNA
            setup_vna = True
        else:
            setup_vna = False

        freqs, mag, phase = return_linear_sweep(
            suffix=suffix, setup_vna=setup_vna, **kwargs)(d)

        m0 = peak_finder(freqs, mag)
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
        return MeasurementFunction(measurement_function, [
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
        return MeasurementFunction(measurement_function, [
            DataParameter(name="resonance_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])


def measure_adaptive_linear_sweep(suffix='', **kwargs):
    """Pysweep VNA measurement function that measures S21 in a window around a
    frequency f0 as be updated through functions such as `measure_resonance_frequency`.
    This is helpful when measuring S21 versus parameters that change the resonance frequency;
    one can first coarsely determine where the resonance is and then finely measure around it.
    Typically one would provide kwargs such as span and npts.

    Args:
    suffix (int): suffix added to the DataParameters.
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
    def measurement_function(d):
        data = return_linear_sweep(
            suffix=suffix, setup_vna=True, center=d["f0"], **kwargs)(d)
        return [data[0], data[1], data[2]]

    return measurement_function



# ---------------------------------- CW functions from here onwards ------------------------------------------


def setup_CW_sweep(
    station, cw_frequency=None, npts=None, t_int=None, bw=None, pwr=None, electrical_delay=None, chan="S21",
):
    """Function that sets up the VNA in CW mode according to the specified parameters, leaving
    other parameters intact. 
    One should choose to specify either t_int or npts, otherwise t_int will be ignored.
    Assumes that a channel with name `chan` is already created. 

    Args:
        station: QCoDeS station that contains a R&S ZNB VNA instrument
        cw_frequency (Hz): frequency at which the CW sweep is performed.
        npts (int): number of points in the CW sweep. Between 1 and 1e5.
        t_int (s): total time the measurement takes (npts/bw)
        bw (Hz): VNA bandwidth
        pwr (dBm): VNA power
        electrical_delay (s): electrical delay used by the VNA. I am not sure if this is required in CW mode.
        chan: name of VNA channel to be used

    """

    vna_trace = getattr(station.vna.channels, chan)
    try:
        # gets you into CW in Jaap's new ZNB class but it might nto yet be implemented
        vna_trace.setup_cw_sweep()
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
    """Pysweep VNA measurement function that returns a CW trace, either given the currently
    set VNA parameters or for a custom set of parameters specified via kwargs when setup_vna=true.
    By setting suffix one can measure several frequencies
    versus a parameter in the same measurement without the parameter names interfering.
    Note that it is not entirely clear to what degree `times` is accurate!

    Args:
        suffix (int): suffix added to the DataParameters.
        setup_vna (boolean): whether to use the current VNA settings or pass new ones via kwargs.
        kwargs: see `setup_CW_sweep`.


    Returns:
    Pysweep measurement function
    """
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
    def measurement_function(d):
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

    return measurement_function


def measure_PSD_averaged(averages=1, suffix='', **kwargs):
    """Pysweep VNA measurement function that returns the average of the PSD of the I and Q quadratures. 
    Uses a rolling average, so at most two traces exist in the memory simultaneously. Averages can thus be large without memory issues.
    Mainly to be used as a diagnostic tool; `frequency` is based on `times` which might not be entirely correct.
    To be implemented: make averaging optional. 

    Args:
        averages (int): number of averages to be performed.
        suffix (int): suffix added to the DataParameters.
        setup_vna (boolean): whether to use the current VNA settings or pass new ones via kwargs.
        kwargs: see `setup_CW_sweep`.

    Returns:
    Pysweep measurement function
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
    def measurement_function(d):
        station = d["STATION"]

<<<<<<< HEAD
        for ii in range(averages):
            vna_data = return_cw_sweep(
                setup_vna=True, cw_frequency=f0, bw=bandwidth, t_int=t_int, npts=points, **kwargs)(d)
=======
        if bool(kwargs):
            setup_CW_sweep(station=station, **kwargs)
        bw = station.vna.S21.bandwidth()
        sweep_time = station.vna.S21.sweep_time()
        npts = station.vna.S21.npts()
        times = np.linspace(1 / bw, sweep_time, npts)
        n = times.size
        step = np.diff(times)[0]

        for ii in range(int(averages)):
            vna_data = return_cw_sweep()(d)
>>>>>>> 9241dbb2f565842f976fed04cc90363c121c98ff
            I = vna_data[1]
            Q = vna_data[2]
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

    return measurement_function


def measure_SNR_CW(suffix='', **kwargs):
    """Pysweep VNA measurement function that measures the mean, std and SNR in dB at a single 1 frequency point. 

    Args:
        suffix (int): suffix added to the DataParameters.
        setup_vna (boolean): whether to use the current VNA settings or pass new ones via kwargs.
        kwargs: see `setup_CW_sweep`.

    Returns:
    Pysweep measurement function
    """
    @MakeMeasurementFunction(
        [
            DataParameter(name="signal" + str(suffix), unit="dB"),
            DataParameter(name="noise" + str(suffix), unit="dB"),
            DataParameter(name="SNR" + str(suffix), unit="dB"),
        ]
    )
    def measurement_function(d):
        if bool(kwargs):  # checks if there are kwargs, otherwise we can skip setting up the VNA
            setup_vna = True
        else:
            setup_vna = False

        vna_data = return_cw_sweep(setup_vna=setup_vna, **kwargs)(d)
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
    return measurement_function


def return_cw_point(suffix='', setup_vna=False, **kwargs):
    """Pysweep VNA measurement function that returns an averaged CW point, either given the currently
    set VNA parameters or for a custom set of parameters specified via kwargs when setup_vna=true.
    By setting suffix one can measure several frequencies
    versus a parameter in the same measurement without the parameter names interfering.

    Args:
        suffix (int): suffix added to the DataParameters.
        setup_vna (boolean): whether to use the current VNA settings or pass new ones via kwargs.
        kwargs: see `setup_CW_sweep`.


    Returns:
    Pysweep measurement function
    """
    @MakeMeasurementFunction(
        [DataParameter(name="amplitude" + str(suffix), unit=""),
         DataParameter(name="phase" + str(suffix), unit="rad")]
    )
    def measurement_function(d):
        station = d["STATION"]
        if setup_vna:
            setup_CW_sweep(station=station, **kwargs)

        if not station.vna.rf_power():
            station.vna.rf_on()

        data = list(station.vna.S21.point_fixed_frequency_mag_phase())

        station.vna.rf_off()

        return data
    return measurement_function


def measure_2tone_sweep(frequencies, cw_frequency='dict', qubsrc_power=None, settling_time=10e-6, suffix='', **kwargs):
    """Pysweep VNA measurement function that creates a quasi-hardware sweep for doing two-tone spectroscopy. 
    In essence it combines doing return_cw_point versus a sweep object of frequencies into a single measurement function.
    By creating a dedicated measurement function for this, one can easily wrap it with other functions,
    for example to do adaptive qubit spectroscopy or to find the qubit frequency and pass that on to a subsequent function.
    Think for example of measuring the Rabi frequency versus a gate voltage. 
    Perhaps this construction is not neccesary, but I could not think of a way around it. 
    Furthermore, currently the qubsrc is hardcoded. It would be better to give it as an input. But it gets a bit tricky because
    not every source has the same commands for on, off, modulation, etc. 

    Args:
        frequencies (array, Hz): the frequencies over which to perform the 2tone spectroscopy.
        cw_frequency (str, numeric): the CW frequency at which to perform the measurement.
        When this is 'dict', the value d["f0"] is used. Otherwise the input value is used.
        qubsrc_power (dBm): the power set on the qubsrc.
        settling_time (s): the waiting time after setting the qubsrc to its next point in frequencies.
        suffix (int): suffix added to the DataParameters.
        kwargs: see `setup_CW_sweep`.


    Returns:
    Pysweep measurement function
    """
    def measurement_function(d):
        station = d["STATION"]
        station.qubsrc.output_rf('ON')
        station.qubsrc.modulation_rf('OFF')

        if qubsrc_power != None:
            station.qubsrc.power(qubsrc_power)

        mag = np.zeros_like(frequencies)
        phase = np.zeros_like(frequencies)

        if cw_frequency == 'dict':
            setup_CW_sweep(station=station, cw_frequency=d["f0"], **kwargs)
        elif cw_frequency is not None or bool(kwargs):
            setup_CW_sweep(station=station,
                           cw_frequency=cw_frequency, **kwargs)

        for ii in range(len(frequencies)):
            station.qubsrc.frequency(frequencies[ii])
            time.sleep(settling_time)
            data = return_cw_point()(d)
            mag[ii] = data[0]
            phase[ii] = data[1]

        station.qubsrc.output_rf('OFF')

        return [frequencies, mag, phase]

    return MeasurementFunction(measurement_function, [
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


def measure_qubit_frequency(frequencies, suffix='', save_trace=True, peak_finder=None, **kwargs):
    """Pysweep VNA measurement function that measures a qubit frequency `fq` and stores it in the dictionary, 
    similar to 'measure_resonance_frequency'. 

    Args:
        frequencies (array, Hz): the frequencies over which to perform the 2tone spectroscopy.
        suffix (int): suffix added to the DataParameters.
        save_trace (boolean): whether to save the full VNA trace and the determined f0 or only f0.
        peak_finder: Function that finds a peak from VNA output. See for example general_tools -> peak_finding.py
        kwargs: see `measure_2tone_sweep` and `setup_CW_sweep`.


    Returns:
    Pysweep measurement function
    """
    def measurement_function(d):
        freqs, mag, phase = measure_2tone_sweep(
            frequencies=frequencies, **kwargs)(d)

        m0 = peak_finder(freqs, mag)

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
        return MeasurementFunction(measurement_function, [
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
        return MeasurementFunction(measurement_function, [
            DataParameter(name="qubit_frequency" + str(suffix),
                          unit="Hz",
                          paramtype="numeric",
                          ),
        ])
