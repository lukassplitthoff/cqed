# Automatized optimization of magnetic field alignment based on maximizing
# the frequency of a reference resonator
# cQED group @Kouwenhoven-lab TU Delft

from cqed.custom_pysweep_functions.vna import return_vna_trace
import numpy as np
from time import sleep
import matplotlib.pyplot as plt


class FieldAligner:

    def __init__(self, station=station, f0=None, search_span=25e6,
                 frq_resolution=200e3, max_field_amp=10e-3, wiggle_step=0.4e-3,
                 min_wiggle=0.1e-3, vna_avgs=None, vna_bw=None, vna_pwr=None):

        self.d = {'STATION': station}
        self.vna_trace = getattr(station.vna.channels, chan)
        self.mgnt = station.mgnt

        self.max_amplitude = max_field_amp
        self.wiggle_step = wiggle_step
        self.min_wiggle = min_wiggle

        if vna_avgs is None:
            self.vna_avgs = self.vna_trace.avg()
        else:
            self.vna_avgs = vna_avgs

        if vna_bw is None:
            self.vna_bw = self.vna_trace.bandwidth()
        else:
            self.vna_bw = vna_bw

        if vna_pwr is None:
            self.vna_pwr = self.vna_trace.power()
        else:
            self.vna_pwr = vna_pwr

        self.current_f0 = f0
        self.search_span = search_span
        self.frq_resolution = frq_resolution

        if f0 is None:
            self.current_f0 = self.find_resonance()

        self._target_fcn = self.find_resonance

        self.objectives_meas = []
        self.locations = []
        self.nsteps = 1

    def find_resonance(self):

        self.vna_trace.start(self.current_f0 - self.search_span / 2.)
        self.vna_trace.stop(self.current_f0 + self.search_span / 2.)
        self.vna_trace.npts(int(self.search_span / self.frq_resolution) + 1)
        self.vna_trace.bandwidth(self.vna_bw)
        self.vna_trace.power(self.vna_pwr)
        self.vna_trace.avg(self.vna_avgs)

        data = return_vna_trace(self.d)
        return data[0][np.argmin(data[1])]

    def optimize_x(self):

        """ Wiggle the x-axis of the magnetic field up and down until the
        objective is optimized to the desired accuracy.
        Picture a Roombah, I have been told.
        Inputs:
        max_magnitude (float): maximal magnetic field value for x, unit: T
        wiggle_step (float): initial step for changes in magnetic field, unit: T
        min_wiggle (float): field step taken after which
                                the protocol terminates, unit: T
        observer_fn: a callable which measures the objective
        """

        self.objectives_meas = []
        self.locations = []

        self.nsteps = 1
        self.objectives_meas.append(self.current_f0)
        self.locations.append(self.mgnt.x_measured())

        direction = 1
        wiggle_step = self.wiggle_step

        while wiggle_step > self.min_wiggle:

            pos = self.mgnt.x_measured()
            if abs(pos + direction * wiggle_step) > self.max_amplitude:
                raise Exception('X-axis magnitude limit reached.')

            print('f0 = {0:.4f} GHz'.format(self.current_f0 / 1e9),
                  '{0:.4f} T'.format(self.mgnt.x_measured()), '->',
                  '{0:.4f} T'.format(pos + direction * wiggle_step))

            self.mgnt.x_target(pos + direction * wiggle_step)
            self.mgnt.ramp(mode='safe')
            sleep(1.)
            newpos = self.mgnt.x_measured()

            if np.abs(pos - newpos) < 0.8 * wiggle_step:
                continue

            new_objective = self._target_fcn()
            self.objectives_meas.append(new_objective)
            self.locations.append(self.mgnt.x_measured())
            self.nsteps += 1

            if new_objective < self.current_f0:
                self.current_f0 = new_objective
                direction = -1 * direction

                if direction == 1:
                    # should we round this to 0.1mT?
                    wiggle_step = wiggle_step / 2.
            else:
                self.current_f0 = new_objective

        print('Optimization finished after {} steps'.format(self.nsteps),
              '\nTarget value from {0:.4f} GHz to {0:.4f} GHz'.format(
                  self.objectives_meas[0], self.objectives_meas[-1]),
              '\nX-field = {0:.4f} T'.format(self.mgnt.x_measured()))

    def plot_optimization(self):

        plt.plot(self.locations, self.objectives_meas)
        plt.scatter(self.locations, self.objectives_meas, c=self.objectives_meas)
        plt.colorbar()
