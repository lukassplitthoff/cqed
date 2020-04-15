# Automatized optimization of magnetic field alignment based on maximizing
# the frequency of a reference resonator
# cQED group @Kouwenhoven-lab TU Delft

from cqed.custom_pysweep_functions.vna import return_vna_trace
from cqed.custom_pysweep_functions.magnet import measure_magnet_components_sph
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

#TODO:
# test code
# save vna traces during field optimization using QCoDeS


class FieldAligner:

    def __init__(self, station=None, f0=None, search_span=25e6,
                 frq_resolution=200e3, max_field_amp=10e-3, wiggle_step=0.4e-3,
                 min_wiggle=0.1e-3, theta=0., phi=0., reoptimization_threshold=10e6, 
                 vna_avgs=None, vna_bw=None, vna_pwr=None, chan='S21', x_field_par=None, 
                 current_source=None):

        self.d = {'STATION': station}
        self.vna_trace = getattr(station.vna.channels, chan)
        self.mgnt = station.mgnt
        self.x_field_par = x_field_par
        self.current_source = current_source

        self.max_amplitude = max_field_amp
        self.wiggle_step = wiggle_step
        self.min_wiggle = min_wiggle

        self.theta = theta
        self.phi = phi
        self.reoptimization_threshold = reoptimization_threshold

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

        if f0 is None:
            self.current_f0 = self.find_resonance()
            self.optimized_f0 = self.current_f0
        else:
            self.current_f0 = f0
            self.optimized_f0 = f0
        
        self.search_span = search_span
        self.frq_resolution = frq_resolution
        self._target_fcn = self.find_resonance

        self.objectives_meas = []
        self.locations = []
        self.nsteps = 1

    def find_resonance(self):

        # Preserve VNA setting of potential measurement
        with self.vna_trace.status.set_to(1):
            self.vna_trace.start(int(self.current_f0 - self.search_span / 2.))
            self.vna_trace.stop(int(self.current_f0 + self.search_span / 2.))
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
        self.locations.append(self.x_field_par())

        direction = 1
        wiggle_step = self.wiggle_step

        while wiggle_step > self.min_wiggle:

            pos = self.x_field_par()
            if abs(pos + direction * wiggle_step) > self.max_amplitude:
                raise Exception('X-axis magnitude limit reached.')

            if self.current_source == 'Keysight': #the keysight is not bipolar
                while pos + direction * wiggle_step < 0:
                    print(wiggle_step, pos + direction * wiggle_step, pos)
                    wiggle_step = wiggle_step / 2.
                    wiggle_step = 5e-6*round(float(wiggle_step)/5e-6)

            print('f0 = {0:.4f} GHz, sweeping from '.format(self.current_f0 / 1e9) +
                  '{0:.3f} mT to '.format(self.x_field_par()*1000) +
                  '{0:.3f} mT'.format((pos + direction * wiggle_step)*1000))

            self.x_field_par(pos + direction * wiggle_step)
            sleep(10.)
            
            # if self.current_source = 'Oxford':
            #     self.mgnt.x_target(pos + direction * wiggle_step)
            #     self.mgnt.x_ramp_to_target()
            #     sleep(1.)
            #     newpos = self.mgnt.x_measured()

            # if self.current_source = 'Keysight':



            #I dont think we need this anymore
            # if np.abs(pos - newpos) < 0.8 * wiggle_step:
            #     continue

            new_objective = self._target_fcn()
            self.objectives_meas.append(new_objective)
            self.locations.append(self.x_field_par())
            self.nsteps += 1

            if new_objective < self.current_f0:
                self.current_f0 = new_objective
                direction = -1 * direction
                print('Turning around')

                if direction == 1:
                    # rounding to multiples of 5uT because 0.15mA is the current setting resolution of the mercuryIPS & the coil constant is around 58A/T.
                    wiggle_step = wiggle_step / 2.
                    wiggle_step = 5e-6*round(float(wiggle_step)/5e-6)
                    if wiggle_step>self.min_wiggle:
                        print('Halving step size')
            else:
                self.current_f0 = new_objective

        self.optimized_f0 = self.current_f0

        print('Optimization finished after {} steps'.format(self.nsteps),
              '\nTarget value from {0:.4f} GHz to '.format(self.objectives_meas[0]/1e9)+'{0:.4f} GHz'.format(self.objectives_meas[-1]/1e9),
              '\nFinal X-field = {0:.3f} mT'.format(self.x_field_par()*1000))

    def plot_optimization(self):
        plt_xs = np.array(self.locations)
        plt_ys = np.array(self.objectives_meas)
        plt.plot(plt_xs*1000, plt_ys/1e9)
        plt.scatter(plt_xs*1000, plt_ys/1e9, c=range(len(plt_xs))) #color now encodes order of measurement
        plt.xlabel('X field [mT]')
        plt.ylabel('Resonance frequency [GHz]')

    def r_setter(self, r):
        ''' Ramp to a magnetic field of radius r, starting from the orientation given by theta and phi.
        If the resonance frequency drops by too much, the orientation then gets adjusted by the compensation field. At the end of each run the angles are updated.
         '''     
        current_x = self.x_field_par()
        current_y = self.mgnt.y_measured()
        current_z = self.mgnt.z_measured()
        
        new_x = r*np.sin(np.radians(self.theta))*np.cos(np.radians(self.phi))
        new_y = r*np.sin(np.radians(self.theta))*np.sin(np.radians(self.phi))
        new_z = r*np.cos(np.radians(self.theta))       
        
        if abs(current_x-new_x)> 20e-3:
            raise Exception('x step size too large!')
        if abs(current_z-new_z)> 20e-3:
            raise Exception('z step size too large!')
        if abs(current_y-new_y)> 20e-3:
            raise Exception('y step size too large!')

        print(f"Sweeping x-axis field from {self.x_field_par()} to {new_x}")
        print(f"Sweeping y-axis field from {self.mgnt.y_measured()} to {new_y}")
        print(f"Sweeping z-axis field from {self.mgnt.z_measured()} to {new_z}")

        self.x_field_par(new_x)
        self.mgnt.y_target(new_y)
        self.mgnt.z_target(new_z)
        self.mgnt.ramp(mode='safe')

        print(f"Sweep succesful")

        new_f0 = self._target_fcn()
        scaled_distance = np.abs(self.optimized_f0 - new_f0)
        if scaled_distance < self.reoptimization_threshold:
            self.current_f0 = new_f0
            print(f"Shift within {self.reoptimization_threshold}, not re-optimizing.")
        else:
            print(f"Current and previous objectives differ by {scaled_distance}, thus re-optimizing.")
            self.optimize_x()
        

        x = self.x_field_par()+1e-9 #avoiding dividing by true zero 
        y = self.mgnt.y_measured()+1e-9
        z = self.mgnt.z_measured()+1e-9

        r_meas = np.sqrt(x**2+y**2+z**2)
        phi_meas = np.arctan2(y,x)/np.pi*180
        if phi_meas < 0:
            phi_meas += 360
        theta_meas = np.arccos(z/r_meas)/np.pi*180

        if r_meas > 1e-3:
            self.theta = theta_meas
            self.phi = phi_meas

        return self.x_field_par()