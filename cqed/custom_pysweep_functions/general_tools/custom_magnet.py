from qcodes.instrument.base import Instrument
import numpy as np
from time import sleep

# needs testing!


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
            get_cmd=self.x_source.x_measured(),
        )

        self.add_parameter(
            "x_target",
            unit="T",
            label="x target field",
            get_cmd=self.x_source.x.target(),
            set_cmd=self.x_source.x_target(),
        )

        self.add_parameter(
            "y_measured",
            unit="T",
            label="y measured field",
            get_cmd=self.mercury.x_measured(),
        )

        self.add_parameter(
            "y_target",
            unit="T",
            label="y target field",
            get_cmd=self.mercury.y.target(),
            set_cmd=self.mercury.y_target(),
        )

        self.add_parameter(
            "z_measured",
            unit="T",
            label="z measured field",
            get_cmd=self.mercury.z_measured(),
        )

        self.add_parameter(
            "z_target",
            unit="T",
            label="z target field",
            get_cmd=self.mercury.z.target(),
            set_cmd=self.mercury.z_target(),
        )

    def ramp(self):
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
            if slave == "y" or slave == "z":
                eval(self.mercury.slave.ramp_to_target())
                # now just wait for the ramp to finish
                # (unless we are testing)
                while slave.ramp_status() == "TO SET":
                    sleep(0.1)
            elif slave == "x":
                eval(self.x_source.slave.ramp_to_target())

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
