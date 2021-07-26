
import numpy as np
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import DelegateParameter
from typing import Optional

from functools import partial
from typing import Optional, Union

from qcodes import VisaInstrument, InstrumentChannel
from qcodes.utils.validators import Numbers, Bool, Enum, Ints


def float_round(val: float) -> int:
    """
    Rounds a floating number

    Args:
        val: number to be rounded

    Returns:
        Rounded integer
    """
    return round(float(val))


class GS610Exception(Exception):
    pass



class GS610(VisaInstrument):
    """
    This is the QCoDeS driver for the Yokogawa GS610 voltage and current source.

    ### MADE BY JAAP, CURRENTLY IN TESTING PHASE

    Args:
      name: What this instrument is called locally.
      address: The GPIB address of this instrument
      kwargs: kwargs to be passed to VisaInstrument class
      terminator: read terminator for reads/writes to the instrument.
    """

    def __init__(self, name: str, address: str, terminator: str = "\n",
                 **kwargs) -> None:
        super().__init__(name, address, terminator=terminator, **kwargs)

        self.add_parameter('output',
                           label='Output State',
                           get_cmd=self.state,
                           set_cmd=lambda x: self.on() if x else self.off(),
                           val_mapping={
                               'off': 0,
                               'on': 1,
                           })

        self.add_parameter('source_mode',
                           label='Source Mode',
                           get_cmd=':SOUR:FUNC?',
                           set_cmd=self._set_source_mode,
                           vals=Enum('VOLT', 'CURR'))

        # When getting the mode internally in the driver, look up the mode as
        # recorded by the _cached_mode property, instead of calling source
        # _mode(). This will prevent frequent VISA calls to the instrument.
        # Calling _set_source_mode will change the chased value.
        self._cached_mode = self.source_mode()
       

        # We want to cache the range value so communication with the instrument
        # only happens when the set the range. Getting the range always returns
        # the cached value. This value is adjusted when calling self._set_range.
        self._cached_range_value: Optional[float] = None

        self.add_parameter('voltage_range',
                           label='Voltage Source Range',
                           unit='V',
                           get_cmd=partial(self._get_range, "VOLT"),
                           set_cmd=partial(self._set_range, "VOLT"),
                           vals=Enum(200e-3, 2e0, 12e0, 20e0, 30e0, 60e0, 110e0))

        # The driver is initialized in the volt mode. In this mode we cannot
        # get 'current_range'. Hence the snapshot is excluded.
        self.add_parameter('current_range',
                           label='Current Source Range',
                           unit='I',
                           get_cmd=partial(self._get_range, "CURR"),
                           set_cmd=partial(self._set_range, "CURR"),
                           vals=Enum(20e-6, 200e-6, 2e-3, 20e-3, 200e-3, 500e-3, 1e0, 2e0, 3e0),
                           snapshot_exclude=False
                           )

        # This is changed through the source_mode interface
        self.range = self.voltage_range if self._cached_mode=='VOLT' else self.current_range

        self._auto_range = False
        self.add_parameter('auto_range',
                           label='Auto Range',
                           set_cmd=self._set_auto_range,
                           get_cmd=lambda: self._auto_range,
                           vals=Bool())

        self.add_parameter('voltage',
                           label='Voltage',
                           unit='V',
                           set_cmd=partial(self._get_set_output, "VOLT"),
                           get_cmd=partial(self._get_set_output, "VOLT")
                           )
        # Again, at init we are in "VOLT" mode. Hence, exclude the snapshot of
        # 'current' as instrument does not support this parameter in
        # "VOLT" mode.
        self.add_parameter('current',
                           label='Current',
                           unit='I',
                           set_cmd=partial(self._get_set_output, "CURR"),
                           get_cmd=partial(self._get_set_output, "CURR"),
                           snapshot_exclude=True
                           )

        # This is changed through the source_mode interface
        self.output_level = self.voltage if self._cached_mode == 'VOLT' else self.current
        
        ## Measurement parameters

        self.add_parameter('measured_volt',
                           label='Measured voltage',
                           unit='V',
                           set_cmd=False,
                           get_cmd=partial(self._get_measurement, "VOLT"),
                           snapshot_exclude=True,
                           get_parser=float
                           )
        self.add_parameter('measured_current',
                           label='Measured current',
                           unit='A',
                           set_cmd=False,
                           get_cmd=partial(self._get_measurement, "CURR"),
                           snapshot_exclude=True,
                           get_parser=float
                           )
        
        self.add_parameter("measurement_mode",
                            label='Measurement mode',
                            unit='',
                            vals=Enum("VOLT", "CURR", "RES"),
                            get_cmd =':SENS:FUNC?',
                            set_cmd =':SENS:FUNC {}')
       
        #@todo implement limits. 
        # self.add_parameter('voltage_limit')
        # self.add_parameter('voltage_lower_limit',
        #                 label='Voltage Upper Protection Limit'
        #                    unit='V',
        #                    vals=Ints(1, 30),
        #                    get_cmd=":SOUR:VOLT:PROT?",
        #                    set_cmd=":SOUR:VOLT:PROT {}",
        #                    get_parser=float_round,
        #                    set_parser=int))
        # self.add_parameter('voltage_upper_limit')
        # self.add_parameter('current_limit')
        # self.add_parameter('current_lower_limit')
        # self.add_parameter('current_upper_limit')

        # self.add_parameter('voltage_limit',
        #                    label='Voltage Protection Limit',
        #                    unit='V',
        #                    vals=Ints(1, 30),
        #                    get_cmd=":SOUR:VOLT:PROT?",
        #                    set_cmd=":SOUR:VOLT:PROT {}",
        #                    get_parser=float_round,
        #                    set_parser=int)

        # self.add_parameter('current_limit',
        #                    label='Current Protection Limit',
        #                    unit='I',
        #                    vals=Numbers(1e-3, 200e-3),
        #                    get_cmd=":SOUR:PROT:CURR?",
        #                    set_cmd=":SOUR:PROT:CURR {:.3f}",
        #                    get_parser=float,
        #                    set_parser=float)

        self.add_parameter('four_wire',
                           label='Four Wire Sensing',
                           get_cmd=':SENS:REM?',
                           set_cmd=':SENS:REM {}',
                           val_mapping={
                              'off': 0,
                              'on': 1,
                           })
        # Note: The guard feature can be used to remove common mode noise.
        # Read the manual to see if you would like to use it

        # Return measured line frequency
        self.add_parameter("line_freq",
                           label='Line Frequency',
                           unit="Hz",
                           get_cmd="SYST:LFR?",
                           get_parser=int)

        # Reset function
        self.add_function('reset', call_cmd='*RST')

        self.connect_message()

    def on(self):
        """Turn output on"""
        self.write('OUTPUT 1')

    def off(self):
        """Turn output off"""
        self.write('OUTPUT 0')

    def state(self) -> int:
        """Check state"""
        state = int(self.ask('OUTPUT?'))
        return state

    def ramp_voltage(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the voltage from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Volt
            step: The ramp steps in Volt
            delay: The time between finishing one step and
                starting another in seconds.
        """
        self._assert_mode("VOLT")
        self._ramp_source(ramp_to, step, delay)

    def ramp_current(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the current from the current level to the specified output.

        Args:
            ramp_to: The ramp target in Ampere
            step: The ramp steps in Ampere
            delay: The time between finishing one step and starting
                another in seconds.
        """
        self._assert_mode("CURR")
        self._ramp_source(ramp_to, step, delay)

    def _ramp_source(self, ramp_to: float, step: float, delay: float) -> None:
        """
        Ramp the output from the current level to the specified output

        Args:
            ramp_to: The ramp target in volts/amps
            step: The ramp steps in volts/ampere
            delay: The time between finishing one step and
                starting another in seconds.
        """
        saved_step = self.output_level.step
        saved_inter_delay = self.output_level.inter_delay

        self.output_level.step = step
        self.output_level.inter_delay = delay
        self.output_level(ramp_to)

        self.output_level.step = saved_step
        self.output_level.inter_delay = saved_inter_delay

    def _get_measurement(self, mode="VOLT"):

        if self.measurement_mode() != mode:
            self.measurement_mode(mode)

        return self.ask("READ?")

    def _get_set_output(self, mode: str,
                        output_level: float = None) -> Optional[float]:
        """
        Get or set the output level.

        Args:
            mode: "CURR" or "VOLT"
            output_level: If missing, we assume that we are getting the
                current level. Else we are setting it
        """
        self._assert_mode(mode)
        if output_level is not None:
            self._set_output(output_level)
            return None
        return float(self.ask(f":SOUR:{self._cached_mode}:LEV?"))

    def _set_output(self, output_level: float) -> None:
        """
        Set the output of the instrument.

        Args:
            output_level: output level in Volt or Ampere, depending
                on the current mode.
        """
        auto_enabled = self.auto_range()

        if not auto_enabled:
            self_range = self._cached_range_value
            if self_range is None:
                raise RuntimeError("Trying to set output but not in"
                                   " auto mode and range is unknown.")
        else:
            mode = self._cached_mode
    
        
        # Check that the range hasn't changed
        if not auto_enabled:
            # Update range
            self.range()
            self_range = self._cached_range_value
            if self_range is None:
                raise RuntimeError("Trying to set output but not in"
                                    " auto mode and range is unknown.")
            # If we are still out of range, raise a value error
            if abs(output_level) > abs(self_range):
                raise ValueError("Desired output level not in range"
                                 " [-{self_range:.3}, {self_range:.3}]".
                                 format(self_range=self_range))

        if auto_enabled:
            auto_str = ":AUTO"
        else:
            auto_str = ""
        cmd_str = ":SOUR:{}:LEV{} {:.5e}".format(self._cached_mode, auto_str, output_level)
        self.write(cmd_str)

    
    def _set_auto_range(self, val: bool) -> None:
        """
        Enable/disable auto range.

        Args:
            val: auto range on or off
        """
        self._auto_range = val

    def _assert_mode(self, mode: str) -> None:
        """
        Assert that we are in the correct mode to perform an operation.
        If check is True, we double check the instrument if this check fails.

        Args:
            mode: "CURR" or "VOLT"
        """
        if self._cached_mode != mode:
            raise ValueError("Cannot get/set {} settings while in {} mode".
                             format(mode, self._cached_mode))

    def _set_source_mode(self, mode: str) -> None:
        """
        Set output mode. Also, exclude/include the parameters from snapshot
        depending on the mode. The instrument does not support
        'current', 'current_range' parameters in "VOLT" mode and 'voltage',
        'voltage_range' parameters in "CURR" mode.

        Args:
            mode: "CURR" or "VOLT"

        """
        if self.output() == 'on':
            raise GS610Exception("Cannot switch mode while source is on")

        if mode == "VOLT":
            self.range = self.voltage_range
            self.output_level = self.voltage
            self.voltage_range.snapshot_exclude = False
            self.voltage.snapshot_exclude = False
            self.current_range.snapshot_exclude = True
            self.current.snapshot_exclude = True
        else:
            self.range = self.current_range
            self.output_level = self.current
            self.voltage_range.snapshot_exclude = True
            self.voltage.snapshot_exclude = True
            self.current_range.snapshot_exclude = False
            self.current.snapshot_exclude = False

        self.write("SOUR:FUNC {}".format(mode))
        self._cached_mode = mode

    def _set_range(self, mode: str, output_range: float) -> None:
        """
        Update range

        Args:
            mode: "CURR" or "VOLT"
            output_range: Range to set. For voltage we have the ranges [10e-3,
                100e-3, 1e0, 10e0, 30e0]. For current we have the ranges [1e-3,
                10e-3, 100e-3, 200e-3]. If auto_range = False then setting the
                output can only happen if the set value is smaller then the
                present range.
        """
        self._assert_mode(mode)
        output_range = float(output_range)

        self._cached_range_value = output_range
        self.write(':SOUR:{}:RANG {}'.format(self._cached_mode, output_range))

    def _get_range(self, mode: str) -> float:
        """
        Query the present range.
        Note: we do not return the cached value here to ensure snapshots
        correctly update range. In fact, we update the cached value when
        calling this method.

        Args:
            mode: "CURR" or "VOLT"

        Returns:
            range: For voltage we have the ranges [10e-3, 100e-3, 1e0, 10e0,
                30e0]. For current we have the ranges [1e-3, 10e-3, 100e-3,
                200e-3]. If auto_range = False then setting the output can only
                happen if the set value is smaller then the present range.
        """
        self._assert_mode(mode)
        self._cached_range_value = float(self.ask(f":SOUR:{self._cached_mode}:RANG?"))
        return self._cached_range_value

    
