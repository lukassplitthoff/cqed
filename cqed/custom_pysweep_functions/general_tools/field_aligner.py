# Automatized optimization of magnetic field alignment based on maximizing
# the frequency of a reference resonator
# cQED group @Kouwenhoven-lab TU Delft

from cqed.custom_pysweep_functions.vna import return_vna_trace
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
from warnings import showwarning

# TODO:
# test code
# save vna traces during field optimization using QCoDeS
# make sure it halts when it loses track of the frequency, or instead
# makes the window larger


class FieldAligner(object):
    """Represents the optimization problem of aligning the magnetic field Args:
    instrument: A QCoDeS (meta)instrument that controls a magnetic field.

    Typically the Oxford mercuryIPS or a custom combination of sources.
    objective: A gettable QCoDeS parameter that returns the value to be
    optimized. Typically a resonator frequency.
    """

    def __init__(self, instrument, objective):

        self.magnet = instrument
        self.objective = objective

    def optimize_x(self,
                   initial_step=200e-6,
                   final_step=50e-6,
                   resolution=20e-6,
                   initial_direction=1,
                   max_amplitude=10e-3,
                   waiting_time=3,
                   return_extra=False,
                   plot=False,
                   verbose=False
                   ):
        """At fixed magnitude of y and z field, maximizes the objective by wiggling
        the x-axis of the magnetic field up and down in decreasing to deal with
        hysteresis, a common pitfall encountered when using an exhaustive search.

        Args:
        initial_step (T): The size of the first step in
        x-field that will be made.
        final_step (T): The step size
        below which the optimization procedure will halt.
        resolution (T): The resolution of the magnet source used.
        Steps taken will be rounded to an integer multiple of this number.
        initial_direction: 1 for starting the search in the positive x
        direction, -1 for negative x direction.
        max_amplitude (T): The largest (absolute) value of x-field that can be set.
        Outside of this range the procedure will abort.
        waiting_time (s): Amount of time waited after setting the field before continuing.
        return_extra (boolean): If `True`, this method will return additional
        data as a dictionary.
        plot (boolean): If `True`, produces a plot of the path that this
        method took to find the optimal objective value.
        verbose (boolean): toggles print statements during optimization

        Returns:
        The optimal field value found.
        If `return_extra=True`, this method returns a tuple of the optimal field
        and a dictionary
        containing diagnostic data.

        """
        objectives_meas = []
        locations = []

        nsteps = 1
        direction = initial_direction
        wiggle_step = resolution * \
            round(float(initial_step) / resolution)
        current_objective = self.objective()
        current_pos = self.magnet.x_measured()

        objectives_meas.append(current_objective)
        locations.append(current_pos)

        while wiggle_step >= final_step:

            if abs(
                    current_pos +
                    direction *
                    wiggle_step) > max_amplitude:
                raise Exception(
                    'X-axis magnitude limit reached.')

            if verbose:
                print(
                    'f0 = {0:.4f} GHz, sweeping from '.format(
                        current_objective /
                        1e9) +
                    '{0:.4f} mT to '.format(
                        current_pos *
                        1e3) +
                    '{0:.4f} mT'.format(
                        (current_pos +
                         direction *
                         wiggle_step) *
                        1e3))

            self.magnet.x_target(
                current_pos + direction * wiggle_step)
            self.magnet.x_source.ramp_to_target()
            sleep(waiting_time)

            new_pos = self.magnet.x_measured()
            new_objective = self.objective()

            objectives_meas.append(new_objective)
            locations.append(new_pos)
            nsteps += 1

            if new_objective < current_objective:
                current_objective = new_objective
                current_pos = new_pos
                direction = -1 * direction
                if verbose:
                    print('Turning around')

                if np.sign(direction) == np.sign(
                        initial_direction):
                    wiggle_step = wiggle_step / 2.
                    wiggle_step = resolution * \
                        round(float(wiggle_step) / resolution)
                    if verbose:
                        print('Halving step size')
            else:
                current_objective = new_objective
                current_pos = new_pos

        r_meas, theta_meas, phi_meas = self.magnet_components_sph()
        if r_meas > 2e-3:  # at low fields the angles are poorly defined due to noise
            self.theta = theta_meas
            self.phi = phi_meas

        optimum = objectives_meas[-1]

        print('Optimization finished after {} steps'.format(nsteps),
              '\nTarget value from {0:.4f} GHz to '.format(
                  objectives_meas[0] / 1e9) + '{0:.4f} GHz'.format(optimum / 1e9),
              '\nFinal X-field = {0:.3f} mT'.format(new_pos * 1e3))

        extra = {
            'objectives': objectives_meas,
            'fields': locations
        }

        if plot:
            plt_xs = np.array(extra['fields'])
            plt_ys = np.array(extra['objectives'])
            plt.figure()
            plt.plot(plt_xs * 1e3, plt_ys)
            plt.scatter(plt_xs * 1e3, plt_ys, c=plt_ys)
            plt.colorbar()
            plt.xlabel('X field [mT]')
            plt.ylabel('Objective [GHz')

        if return_extra:
            return optimum, extra
        else:
            return optimum

    def optimize_and_ramp_r(self,
                            rs,
                            initial_theta,
                            initial_phi,
                            optimize_first,
                            optimize_strategy=None,
                            optimize_at=None,
                            reoptimization_threshold=None,
                            waiting_time=3,
                            max_field_strength=1.5,
                            return_extra=True,
                            verbose=True,
                            **kwargs
                            ):
        """Ramps the magnetic field over the magnitude values provided, optimizing
        the objective by adjusting the x-field whenever required as given by
        `optimize_strategy`. Use the **kwargs to set up optimize_x!

        Args: rs (T): Array of magnitude values along which the field is
        ramped. Smaller steps implies less risk of losing resonances but takes
        more time.
        initial_theta (deg): intial angle theta used to set the
        direction of the field.
        initial_phi (deg): intial angle phi used to set
        the direction of the field.
        optimize_first (boolean): Whether to
        optimize the field before starting the ramp procedure. Note that this
        might overwrite the input values of theta and phi!
        optimize_strategy (str): three options are implemented:
        `objective_decrease`, for which one optimizes the alignment if the objective
        has decreased by more than `reoptimization_threshold` since the last
        call to `optimize_x`. The next option is `optimize_at_fields`,
        for which the alignment is optimized only at the field given by `optimize_at`.
        And finally, if `None` is chosen, the alignment is never optimized.
        optimize_at (T): Array of values at which the alignment will be optimized if
        `optimize_strategy` = `optimize_at`
        reoptimization_threshold (GHz): Value against which the change in objective is
        compared in order to establish if the field has to be aligned if
        `optimize_strategy` = `objective_decrease`.
        waiting_time (s): Amount of time waited after
        sweeping the field before continuing.
        return_extra (boolean): If `True`, this method will return additional
        data as a dictionary.
        verbose (boolean): toggles print statements during optimization and
        sweeping.
        Returns:     The optimal field value found.     If
        `return_extra=True`, this method returns a tuple of the
        optimal field and a dictionary containing diagnostic data.

        """

        if max_field_strength > 1.5:
            showwarning(
                'Be aware that mu-metal shields are saturated by too large magnetic fields and will not work afterwards. '
                'Are you sure you want to go to more than 1.5 T?',
                ResourceWarning,
                'cqed/cqed/custom_pysweep_functions/field_aligner',
                193)
        if optimize_strategy == 'objective_decrease':
            if reoptimization_threshold is None:
                raise Exception(
                    'You need to specify a reoptimization_threshold when using the `objective_decrease` option.')
        elif optimize_strategy == 'optimize_at_fields':
            if optimize_at is None:
                raise Exception(
                    'You need to specify at which fields to optimize when using the `optimize_at_fields` option.')

        extra = {}
        objective_history = []
        optima_history = []

        self.theta = initial_theta
        self.phi = initial_phi

        if optimize_first:
            current_objective = self.optimize_x(
                verbose=verbose,
                return_extra=False,
                waiting_time=waiting_time,
                **kwargs)
        else:
            current_objective = self.objective()

        last_optimized = current_objective

        for r in rs:
            current_x = self.magnet.x_measured()
            current_y = self.magnet.y_measured()
            current_z = self.magnet.z_measured()

            objective_history.append(current_objective)
            optima_history.append([self.theta, self.phi])

            new_x = r * np.sin(np.radians(self.theta)) * \
                np.cos(np.radians(self.phi))
            new_y = r * np.sin(np.radians(self.theta)) * \
                np.sin(np.radians(self.phi))
            new_z = r * np.cos(np.radians(self.theta))

            if np.sqrt(
                    new_x**2 +
                    new_y**2 +
                    new_z**2) > max_field_strength:
                raise Exception(
                    'Target field exceeds max field strength.')

            if verbose:
                print(
                    f"Sweeping x-axis field from {current_x} to {new_x}")
                print(
                    f"Sweeping y-axis field from {current_y} to {new_y}")
                print(
                    f"Sweeping z-axis field from {current_z} to {new_z}")

            self.magnet.x_target(new_x)
            self.magnet.y_target(new_y)
            self.magnet.z_target(new_z)
            sleep(waiting_time)

            if verbose:
                print(f"Sweep finished.")

            new_objective = self.objective()
            objective_history.append(new_objective)

            if optimize_strategy == 'objective_decrease':
                distance = last_optimized - new_objective
                if distance < reoptimization_threshold:
                    current_objective = new_objective
                    print(
                        f"Shift within {reoptimization_threshold}, not re-optimizing.")
                else:
                    print(
                        f"Current and previous objectives differ by {distance}, thus re-optimizing.")
                    current_objective = self.optimize_x(
                        verbose=verbose,
                        return_extra=False,
                        waiting_time=waiting_time,
                        **kwargs)
                    last_optimized = current_objective

            elif optimize_strategy == 'optimize_at_fields':
                if np.any(
                    np.isclose(
                        r,
                        optimize_at,
                        atol=np.diff(rs)[0] /
                        2)):
                    print(
                        f"Current field is in `optimize_at`, thus re-optimizing.")
                    current_objective = self.optimize_x(
                        verbose=verbose,
                        return_extra=False,
                        waiting_time=waiting_time,
                        **kwargs)
                else:
                    current_objective = new_objective

            objective_history.append(current_objective)
            optima_history.append([self.theta, self.phi])

            if return_extra:
                extra['history'] = {
                    'rs': rs,
                    'objectives': objective_history,
                    'optima': optima_history
                }
                return objective_history[-1], extra
            else:
                return objective_history[-1]

    def magnet_components_sph(self):
        """Return the x, y, z component of the magnet, convert to spherical using the
        ISO 80000-2:2009 physics convention for the (r, theta, phi)

        <--> (x, y, z) definition and return as list.

        """
        x = self.magnet.x_measured() + \
            1e-9  # avoiding dividing by true zero
        y = self.magnet.y_measured() + 1e-9
        z = self.magnet.z_measured() + 1e-9
        
        r_meas = np.sqrt(x**2 + y**2 + z**2)
        phi_meas = np.arctan2(y, x) / np.pi * 180
        if phi_meas < 0:
            phi_meas += 360  # convention is 0 to 360 not -180 to 180
        theta_meas = np.arccos(z / r_meas) / np.pi * 180

        return [r_meas, theta_meas, phi_meas]
