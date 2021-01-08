import numpy as np
import scipy.constants as pyc
from scipy.optimize import fsolve
from xarray import DataArray, merge


class AbsSquid:
    """ A class to calculate parameters of the SQUID geometry in the Andreev molecule """

    def __init__(self, Phi_L, Phi_R, I_L, I_R, L_m, n_L, n_R):
        """
        @param Phi_L:
        @param Phi_R:
        @param L_m:
        @param n_L:
        @param n_R:
        """

        self.Phi_L = Phi_L
        self.Phi_R = Phi_R
        self.I_L = I_L
        self.I_R = I_R
        self.L_m = L_m
        self.n_L = n_L
        self.n_R = n_R

        self.phi_R = np.zeros((len(Phi_L), len(Phi_R)))
        # self.phi_R = np.zeros_like(Phi_R)
        self.phi_L = np.zeros_like(self.phi_R)

        self._reduced_flxqntm = pyc.h / 2. / pyc.e / 2. / pyc.pi

    def _cpr(self, i_c, phase):
        """ Current phase relation of the nanowire junction, for now assumed to be sinusoidal
        @param i_c:
        @param phase:
        @return:
        """
        return i_c * np.sin(phase)

    def _phi_L(self, phi_R, Phi_L, Phi_R, n_L, n_R):
        """
        Return the phase drop across the left junction as a function of external fluxes and the phase drop across the
        right junction.
        @param phi_R:
        @param Phi_L:
        @param Phi_R:
        @param n_L:
        @param n_R:
        @return:
        """
        return (Phi_L + Phi_R) / self._reduced_flxqntm - phi_R - 2. * pyc.pi * (n_L + n_R)

    def _transcendental_phi_R(self, phi_R, Phi_L, Phi_R, I_L, I_R, L_m, n_L, n_R):
        """
        Calculate the phase drop across the right junction as a function of the external fluxes, shunting inductance,
        and trapped fluxes.
        @param Phi_L:
        @param Phi_R:
        @param L_m:
        @param n_L:
        @param n_R:
        @return:
        """
        phi_L = self._phi_L(phi_R, Phi_L, Phi_R, n_L, n_R)
        I_m = self._cpr(I_R, phi_R) + self._cpr(I_L, phi_L)

        return 2.*pyc.pi * (n_L-n_R) - 1./self._reduced_flxqntm * (Phi_L-Phi_R) + phi_L - phi_R \
               + 2. * L_m / self._reduced_flxqntm * I_m

    def _solve_phi_R(self):
        """
        Solve the transcendental equation
        @return:
        """
        guess = pyc.pi

        for i, _Phi_L in enumerate(self.Phi_L):

            for j, _Phi_R in enumerate(self.Phi_R):

                sol = fsolve(self._transcendental_phi_R, guess, (_Phi_L, _Phi_R, self.I_L, self.I_R, self.L_m,
                                                                 self.n_L, self.n_R))
                self.phi_R[i, j] = sol
                self.phi_L[i, j] = self._phi_L(sol, _Phi_L, _Phi_R, self.n_L, self.n_R)
                guess = sol


    def _Ic_max(self, I_L, I_R, phi_L, phi_R):

        return np.sqrt(I_L**2 + I_R**2 + 2 * I_L * I_R * np.cos(phi_L - phi_R))


    def return_phases(self):

        self._solve_phi_R()
        phi_Ls = DataArray(self.phi_L, coords={'Phi_L': self.Phi_L, 'Phi_R': self.Phi_R},
                           dims=['Phi_L', 'Phi_R'], name='phi_L')
        phi_Rs = DataArray(self.phi_R, coords={'Phi_L': self.Phi_L, 'Phi_R': self.Phi_R},
                           dims=['Phi_L', 'Phi_R'], name='phi_R')

        # phi_Ls = DataArray(self.phi_L, coords={'Phi': self.Phi_L},
        #                    dims=['Phi'], name='phi_L')
        # phi_Rs = DataArray(self.phi_R, coords={'Phi': self.Phi_L},
        #                    dims=['Phi'], name='phi_R')

        return merge([phi_Ls, phi_Rs])

    def phases_vs_field(self, B, A_l, A_r):

        phi_l = np.zeros_like(B)
        phi_r = np.zeros_like(B)

        guess = np.pi
        for i in range(len(B)):
            if i > 0:
                guess = phi_r[i-1]
            sol = fsolve(self._transcendental_phi_R, guess, (B[i]*A_l, B[i]*A_r, self.I_L, self.I_R, self.L_m,
                                                                 self.n_L, self.n_R))
            phi_r[i] = sol
            phi_l[i] = self._phi_L(sol, B[i]*A_l, B[i]*A_r, self.n_L, self.n_R)

        return phi_l, phi_r

    def josephson_inductance(self, i_c, phi):

        return self._reduced_flxqntm / (i_c * np.cos(phi))


    def squid_inductance(self, B, A_l, A_r, I_L, I_R, Lm):

        phi_l, phi_r = self.phases_vs_field(B, A_l, A_r)
        L_l = self.josephson_inductance(I_L, phi_l)
        L_r = self.josephson_inductance(I_R, phi_r)

        return 1. / (1. / Lm + 1. / L_l + 1. / L_r)

    def resonant_freq(self, B, B_offs, f0, Lres, A_l, A_r, I_L, I_R, Lm):

        return f0 / np.sqrt(1. + self.squid_inductance(B + B_offs, A_l, A_r, I_L, I_R, Lm) / Lres)

