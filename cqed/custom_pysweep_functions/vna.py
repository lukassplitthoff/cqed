# Authors: Lukas Splitthoff, Lukas Gruenhaupt, Arno Bargerbos @TUDelft
# 04-DEC-2019

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
    """Pysweep VNA measurement function.

    Returns VNA frequency axis in Hz, linear amplitude in ? and phase in
    radians. Keeps currently set VNA parameters.

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
    fun_str = ""

    for i, c in enumerate(freq_list):
        s = span_list[i]
        fun_str += "cvna.transmission_vs_frequency_explicit({}, {}, suffix={})+".format(
            c, s, i
        )
    fun_str = fun_str[:-1]
    return fun_str
