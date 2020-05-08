from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
import numpy as np
from time import sleep


class CustomGS210(Instrument):
    """Meta instrument that wraps the Yokogawa GS210 to allow setting (x) fields
    rather than currents.

    x-field is hardcoded to allow for easy integrating into CustomMagnet below,
    but we could/should generalize this.

    """

    def __init__(self, name, instrument, coil_constant, step=1e-6, delay=10e-3):
        super().__init__(name)

        self.source = instrument
        self.coil_constant = coil_constant  # A/T
        self.step = step
        self.delay = delay

        self.add_parameter(
            "x_measured", unit="T", label="x measured field", get_cmd=self._get_field,
        )

        self.add_parameter(
            "x_target",
            unit="T",
            label="x target field",
            get_cmd=self._get_target,
            set_cmd=self._set_target,
        )

        self._field_target = self.x_measured()

    def _get_field(self):
        if self.source.output() == "off":
            B_value = 0
        else:
            I_value = self.source.current()
            B_value = I_value / self.coil_constant
        return B_value

    def _get_target(self):
        return self._field_target

    def _set_target(self, val):
        self._field_target = val

    def ramp_to_target(self):
        if self.source.output() != "on":
            self.source.source_mode("CURR")
        self.source.output("on")
        self.source.ramp_current(
            ramp_to=self.x_target()*self.coil_constant, step=self.step, delay=self.delay
        )


class CustomE36313A(Instrument):
    """Meta instrument that wraps the Keysight E36313A to allow setting (x) fields
    rather than currents.

    x-field is hardcoded to allow for easy integrating into CustomMagnet below,
    but we could/should generalize this.

    Need to add inputs
    """

    def __init__(self, name, instrument, coil_constant):
        super().__init__(name)

        self.source = instrument
        self.coil_constant = coil_constant  # A/T

        self.add_parameter(
            "x_measured", unit="T", label="x measured field", get_cmd=self._get_field,
        )

        self.add_parameter(
            "x_target",
            unit="T",
            label="x target field",
            get_cmd=self._get_target,
            set_cmd=self._set_target,
        )

        self._field_target = self.x_measured()

    def _get_field(self):
        if self.source.ch1.enable() == "off":
            B_value = 0
        else:
            I_value = self.source.ch1.current()
            B_value = I_value / self.coil_constant
        return B_value

    def _get_target(self):
        return self._field_target

    def _set_target(self, val):
        self._field_target = val

    def ramp_to_target(self):
        self.source.ch1.enable("on")
        self.source.ch1.source_current(self.x_target()*self.coil_constant)


class CustomMagnet(Instrument):
    """
    Meta instrument to control the magnet using the MercuryIPS for the y and z axes
    and a second source for the x axis.
    """

    def __init__(self, name, mercury_IPS, x_source):
        super().__init__(name)

        self.mercury = mercury_IPS
        self.x_source = x_source

        self.add_parameter(
            "x_measured",
            unit="T",
            label="x measured field",
            get_cmd=self.x_source.x_measured,
        )

        self.add_parameter(
            "x_target",
            unit="T",
            label="x target field",
            get_cmd=self.x_source.x_target,
            set_cmd=self.x_source.x_target,
        )

        self.add_parameter(
            "y_measured",
            unit="T",
            label="y measured field",
            get_cmd=self.mercury.y_measured,
        )

        self.add_parameter(
            "y_target",
            unit="T",
            label="y target field",
            get_cmd=self.mercury.y_target,
            set_cmd=self.mercury.y_target,
        )

        self.add_parameter(
            "z_measured",
            unit="T",
            label="z measured field",
            get_cmd=self.mercury.z_measured,
        )

        self.add_parameter(
            "z_target",
            unit="T",
            label="z target field",
            get_cmd=self.mercury.z_target,
            set_cmd=self.mercury.z_target,
        )

    def ramp(self, mode='safe'):
        """Ramp the fields to their present target value. Mode is always 'safe' for
        now. Did not implement the others.

        In 'safe' mode, the fields are ramped one-by-one in a blocking way that
        ensures that the total field stays within the safe region (provided that
        this region is convex).

        """
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        order = np.argsort(np.abs(np.array(targ_vals) - np.array(meas_vals)))

        for slave in np.array(["X", "Y", "Z"])[order]:
            if slave == "Y" or slave == "Z":
                eval(f"self.mercury.GRP{slave}.ramp_to_target()")
                while eval(f"self.mercury.GRP{slave}.ramp_status()") == "TO SET":
                    sleep(0.1)
            elif slave == "X":
                eval(f"self.x_source.ramp_to_target()")
                #need to figure out of the other sources also have a ramp status, and if so implement that in their custom drivers!

        self._update_field()


    def x_ramp(self):
        #need to look up syntax in oxford IPS
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.x_source.ramp_to_target()
        self._update_field()

    def y_ramp(self):
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.mercury.GRPY.ramp_to_target()
        while self.mercury.GRPY.ramp_status() == "TO SET":
            sleep(0.1)
        self._update_field()

    def z_ramp(self):
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.mercury.GRPZ.ramp_to_target()
        while self.mercury.GRPZ.ramp_status() == "TO SET":
            sleep(0.1)
        self._update_field()   

    def _get_measured(self):
        field_components = []
        for axis in ["x", "y", "z"]:
            field_components.append(eval("self.{0:}_measured()".format(axis)))
        return field_components

    def _get_targets(self):
        field_components = []
        for axis in ["x", "y", "z"]:
            field_components.append(eval("self.{0:}_target()".format(axis)))
        return field_components

    def _update_field(self):
        coords = ["x", "y", "z"]
        [getattr(self, f"{i}_measured").get() for i in coords]
