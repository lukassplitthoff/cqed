from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
import numpy as np
from time import sleep


class CustomGS210(Instrument):
    """Meta instrument that wraps the Yokogawa GS210 to allow setting fields
    rather than currents.

    The parameter structure follows that of the mercuryIPS for easy integration, 
    but perhaps we can/should generalize this somehow. Need to have a look at the 
    AMI driver.

    args:
    coil_constant (A/T): coil_constant of magnet used
    step (A): step size taken when ramping. Note the unit!
    delay (s): delay after each step taken when ramping
    """

    def __init__(self, name, instrument, coil_constant, step=1e-5, delay=5e-2):
        super().__init__(name)

        self.source = instrument
        self.coil_constant = coil_constant  # A/T
        self.step = step
        self.delay = delay

        self.add_parameter(
            "field", unit="T", label="measured field", get_cmd=self._get_field,
        )

        self.add_parameter(
            "field_target",
            unit="T",
            label="target field",
            get_cmd=self._get_target,
            set_cmd=self._set_target,
        )

        self._field_target = self.field()

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
            ramp_to=self.field_target()*self.coil_constant, step=self.step, delay=self.delay
        )


class CustomE36313A(Instrument):
    """Meta instrument that wraps the Keysight E36313A to allow setting fields
    rather than currents.

    args:
    coil_constant (A/T): coil_constant of magnet used
    step (A): step size taken when ramping. Note the unit!
    delay (s): delay after each step taken when ramping
    """

    def __init__(self, name, instrument, coil_constant, step=6e-4, delay=5e-2):
        super().__init__(name)

        self.source = instrument
        self.coil_constant = coil_constant  # A/T
        assert step >= 6e-4 #resolution of the source
        self.step = step
        self.delay = delay

        self.add_parameter(
            "field", unit="T", label="measured field", get_cmd=self._get_field,
        )

        self.add_parameter(
            "field_target",
            unit="T",
            label="target field",
            get_cmd=self._get_target,
            set_cmd=self._set_target,
        )

        self._field_target = self.field()

    def _get_field(self):
        if self.source.ch1.enable() == "off":
            B_value = 0
        else:
            I_value = self.source.ch1.source_current()
            B_value = I_value / self.coil_constant
        return B_value

    def _get_target(self):
        return self._field_target

    def _set_target(self, val):
        assert 10/self.coil_constant > val >= 1e-3/self.coil_constant #ranges of the source
        self._field_target = val

    def ramp_to_target(self):
        #this should be implemented on the driver level! Make a fork and pull request to qcodes_drivers_contrib?
        self.source.ch1.enable("on")
        resolution = 6e-4/self.coil_constant #resolution of the source, which is quite coarse so we hardcode it to avoid weirdness
        step_sign = np.sign(self.field_target()-self.field())
        try:
            field_values = np.append(np.arange(self.field(), self.field_target(), step_sign*self.step), self.field_target())
            field_values = resolution * np.round(field_values / resolution)
            for val in field_values:
                self.source.ch1.source_current(val*self.coil_constant)
                sleep(self.delay)
        except:
            pass
            #hacky workaround for when your higher level sweep function starts from the current setpoint



class CustomMagnet(Instrument):
    """
    Meta instrument to control the magnet using three individual magnet controllers.
    For example: x_source is an AMIGS200, y_source is mgnt.GRPY, z_source is mgnt.GRPZ.
    """

    def __init__(self, name, x_source, y_source, z_source):
        super().__init__(name)

        self.x_source = x_source
        self.y_source = y_source
        self.z_source = z_source

        self.add_parameter(
            "x_measured",
            unit="T",
            label="x measured field",
            get_cmd=self.x_source.field,
        )

        self.add_parameter(
            "x_target",
            unit="T",
            label="x target field",
            get_cmd=self.x_source.field_target,
            set_cmd=self.x_source.field_target,
        )

        self.add_parameter(
            "y_measured",
            unit="T",
            label="y measured field",
            get_cmd=self.y_source.field,
        )

        self.add_parameter(
            "y_target",
            unit="T",
            label="y target field",
            get_cmd=self.y_source.field_target,
            set_cmd=self.y_source.field_target,
        )

        self.add_parameter(
            "z_measured",
            unit="T",
            label="z measured field",
            get_cmd=self.z_source.field,
        )

        self.add_parameter(
            "z_target",
            unit="T",
            label="z target field",
            get_cmd=self.z_source.field_target,
            set_cmd=self.z_source.field_target,
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

        for slave in np.array(["x", "y", "z"])[order]:
            eval(f"self.{slave}_source.ramp_to_target()")
            try: #these try statements are a bit hacky; the oxfordIPS has a ramp status while it is ramping, the AMI/Keysight don't seem to have one, so I just implement it like this
                while eval(f"self.{slave}_source.ramp_status()") == "TO SET":
                    sleep(0.1)
            except:
                pass
        self._update_field()


    def x_ramp(self):
        #need to look up syntax in oxford IPS
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.x_source.ramp_to_target()
        try:
            while self.x_source.ramp_status() == "TO SET":
                sleep(0.1)
        except:
            pass
        self._update_field()

    def y_ramp(self):
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.y_source.ramp_to_target()
        try:
            while self.y_source.ramp_status() == "TO SET":
                sleep(0.1)
        except:
            pass
        self._update_field()

    def z_ramp(self):
        meas_vals = self._get_measured()
        targ_vals = self._get_targets()
        self.z_source.ramp_to_target()
        try:
            while self.z_source.ramp_status() == "TO SET":
                sleep(0.1)
        except:
            pass
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
