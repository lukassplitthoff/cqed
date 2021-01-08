import numpy as np
from scipy.optimize import curve_fit
from scipy.linalg import eig
import matplotlib.pyplot as plt

class QntmJumpTrace:


    def __init__(self, data, timestep=1., complex=True):
        """
        Class to analyse quantum jumps traces and extract transition
        rates between the identified states

        @param data: either a complex 1d-array, or a tuple (real, imag)
        @param complex: bool, default=True
        """

        if complex:
            self.raw_data = data
        else:
            self.raw_data = data[0] + 1.j * data[1]

        self.timestep = timestep
        self.raw_data_rot = np.empty_like(self.raw_data)
        self.raw_hist = []
        self.state_vec = np.empty_like(self.raw_data)
        self.dwell_l = []
        self.dwell_h = []
        self.hist_dwell_l = []
        self.hist_dwell_h = []

        self.fitresult_gauss = None

    def _max_variance_angle(self, data):
        """
        Function from Gijs that rotates data towards max variance
        in x axis.
        """
        i = np.real(data)
        q = np.imag(data)
        cov = np.cov(i, q)
        a = eig(cov)
        eigvecs = a[1]

        if a[0][1] > a[0][0]:
            eigvec1 = eigvecs[:, 0]
        else:
            eigvec1 = eigvecs[:, 1]

        theta = np.arctan(eigvec1[0] / eigvec1[1])
        return theta

    def _rotate_data(self, data):
        """
        Rotate the complex data by a given angle theta
        @param data: complex IQ data to be returned
        @param theta: rotation angle (radians)
        @return: rotated complex array
        """

        theta = self._max_variance_angle(data)
        return data * np.exp(1.j * theta)

    def _plot_IQ_hist(self, data, mtplib_axis, n_bins=50):
        """
        Plot a 2d histogram
        @param figsize:
        @param n_bins:
        @return:
        """
        hist = mtplib_axis.hist2d(data.real, data.imag, bins=n_bins)

    def _dbl_gaussian(self, x, c1, mu1, sg1, c2, mu2, sg2):
        """
        A double gaussian distribution
        @param x:
        @param c1:
        @param mu1:
        @param sg1:
        @param c2:
        @param mu2:
        @param sg2:
        @return:
        """
        res = c1 * np.exp(-(x - mu1) ** 2. / (2. * sg1 ** 2.)) + c2 * np.exp(-(x - mu2) ** 2. / (2. * sg2 ** 2.))
        return res

    def _create_hist(self, data, n_bins):
        """Calculate the histogram and return an x-axis that is of same length as the numbers
        the x-axis represents the middle of the bins"""

        n, bins = np.histogram(data, bins=n_bins, density=True)
        x_axis = bins[:-1] + (bins[1] - bins[0]) / 2.

        return x_axis, n

    def _fit_dbl_gaussian(self, data, n_bins=100, start_param_guess=None):
        """
        Fittin a double gaussian.
        @param n_bins:
        @return:
        """

        I_ax, n = self._create_hist(data, n_bins)

        if start_param_guess is None:
            p_start = [n[20], I_ax[10], I_ax[10] - I_ax[0], n[-20], I_ax[-10], I_ax[10] - I_ax[0]]

        fit, conv = curve_fit(self._dbl_gaussian, I_ax, n, p0=p_start)

        # ensure that the negative gaussian is the first set of data
        if fit[4] < fit[1]:
            fit = np.array([*fit[3:], *fit[:3]])

        return fit

    def _latching_filter(self, arr, means, sigm):

        state = np.ones_like(arr) * 0.5
        dwell_g = []
        dwell_e = []

        # first point is a simple thresholding
        thres = np.mean(means)
        if arr[0] < thres:
            state[0] = 0
        else:
            state[0] = 1

        count_g = 1
        count_e = 1
        for i in range(1, len(arr)):

            if state[i - 1] == 0:
                if arr[i] > means[1] - sigm[1]:
                    state[i] = 1
                    dwell_g.append(count_g)
                    count_g = 1
                else:
                    state[i] = 0
                    count_g += 1

            if state[i - 1] == 1:
                if arr[i] < means[0] + sigm[0]:
                    state[i] = 0
                    dwell_e.append(count_e)
                    count_e = 1
                else:
                    state[i] = 1
                    count_e += 1

        return state, np.array(dwell_g), np.array(dwell_e)

    def _exp_dist(self, x, gamma, a):
        """ Exponential distribution with rate gamma gamma * exp(-gamma * x)
        @param x:
        @param gamma:
        @return:
        """
        return a * np.exp(-gamma * x)

    def _lin_func(self, x, m, c):

        return -m * x + c

    def latching_pipeline(self, n_bins):
        """
        Perform the entire data analysis from raw IQ trace to transition rates between two states
        @return: class instance itself
        """

        self.raw_data_rot = self._rotate_data(self.raw_data)
        self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_data_rot.real)
        self.raw_hist = self._create_hist(self.raw_data_rot.real, n_bins)
        self.state_vec, self.dwell_l, self.dwell_h = self._latching_filter(self.raw_data_rot.real,
                                                                           self.fitresult_gauss[[1, 4]],
                                                                           self.fitresult_gauss[[2, 5]])
        self.hist_dwell_l = self._create_hist(self.dwell_l, np.max(self.dwell_l))
        self.hist_dwell_h = self._create_hist(self.dwell_h, np.max(self.dwell_h))

        # self.rate_lh = curve_fit(self._lin_func, self.hist_dwell_l[0][10:80], np.log(self.hist_dwell_l[1][10:80]),
        #                          p0=[0.01, 1])
        # self.rate_hl = curve_fit(self._lin_func, self.hist_dwell_h[0][10:80], np.log(self.hist_dwell_h[1][10:80]),
        #                          p0=[0.01, 1])

        return self

    def plot_analysis(self, figsize=(16, 9)):

        fig = plt.figure(1, figsize=figsize)
        plt.gcf()
        dim_ax_raw = [0., 0.7, 0.2, 0.27]
        dim_ax_rot = [0., 0.35, 0.2, 0.27]
        dim_ax_dwell_hist = [0., 0., 0.2, 0.27]
        dim_ax_trace = [0.31, 0.55, 0.59, 0.4]
        dim_ax_histy = [0.91, 0.55, 0.09, 0.4]
        dim_ax_state_assign = [0.31, 0.05, 0.69, 0.4]

        ax_raw = plt.axes(dim_ax_raw)
        ax_rot = plt.axes(dim_ax_rot)
        ax_dwell_hist = plt.axes(dim_ax_dwell_hist)
        ax_trace = plt.axes(dim_ax_trace)
        ax_histy = plt.axes(dim_ax_histy)
        ax_state_assign = plt.axes(dim_ax_state_assign)

        ax_raw.hist2d(self.raw_data.real, self.raw_data.imag, bins=50)
        ax_raw.axis('equal')
        ax_raw.set_title('raw data')
        ax_raw.set_xlabel('I (arb. un.)')
        ax_raw.set_ylabel('Q (arb. un.)')

        ax_rot.hist2d(self.raw_data_rot.real, self.raw_data_rot.imag, bins=50)
        ax_rot.axis('equal')
        ax_rot.set_title('rotated data')
        ax_rot.set_xlabel('I (arb. un.)')
        ax_rot.set_ylabel('Q (arb. un.)')

        ax_dwell_hist.plot(self.hist_dwell_l[0], self.hist_dwell_l[1], label='low state')
        # ax_dwell_hist.plot(self.hist_dwell_l[0], self._exp_dist(self.hist_dwell_l[0], self.rate_lh[0][0], np.exp(self.rate_lh[0][1])), 'k')
        ax_dwell_hist.plot(self.hist_dwell_h[0], self.hist_dwell_h[1], label='higher state')
        # ax_dwell_hist.plot(self.hist_dwell_h[0], self._exp_dist(self.hist_dwell_h[0], self.rate_hl[0][0], np.exp(self.rate_hl[0][1])), 'k')

        ax_dwell_hist.set_xlim(1, np.max([np.max(self.hist_dwell_l[0]), np.max(self.hist_dwell_h[0])]))
        ax_dwell_hist.set_yscale('log')
        ax_dwell_hist.set_title('histogram of dwell times')
        ax_dwell_hist.set_ylabel('norm. occurence')
        ax_dwell_hist.set_xlabel('dwell time')
        ax_dwell_hist.legend()

        ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss[1] - self.fitresult_gauss[2],
                              self.fitresult_gauss[1] + self.fitresult_gauss[2], alpha=0.4, color='tab:orange')
        ax_trace.axhline(self.fitresult_gauss[1], ls='dashed', color='tab:orange')
        ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss[4] - self.fitresult_gauss[5],
                              self.fitresult_gauss[4] + self.fitresult_gauss[5], alpha=0.4, color='tab:green')
        ax_trace.axhline(self.fitresult_gauss[4], ls='dashed', color='tab:green')
        ax_trace.plot(np.arange(0, 400, 1), self.raw_data_rot.real[:400], 'k.-', label='real rotated data')
        ax_trace.set_xlim(-1, 401)
        ax_trace.set_ylim(self.fitresult_gauss[1] - 3 * self.fitresult_gauss[2],
                          self.fitresult_gauss[4] + 3 * self.fitresult_gauss[5])
        ax_trace.set_title('rotated data trace excerpt')
        ax_trace.set_xlabel('timestep')
        ax_trace.set_ylabel('I (arb. un.)')
        ax_histy.barh(self.raw_hist[0], self.raw_hist[1], (self.raw_hist[0][1] - self.raw_hist[0][0]) * 0.8,
                      label='rot. data')
        ax_histy.plot(self._dbl_gaussian(self.raw_hist[0], *self.fitresult_gauss), self.raw_hist[0], 'k', label='fit')
        ax_histy.tick_params(direction='in', labelleft=False)
        ax_histy.set_ylim(ax_trace.get_ylim())
        ax_histy.set_xlabel('norm. occurence')
        ax_histy.set_title('histogram rot. data')
        ax_histy.legend()


        ax_state_assign.plot(np.arange(0,500,1), self.raw_data_rot.real[:500], 'k.-', label='real rotated data')
        ax_state_assign.plot(np.arange(0,500,1), (self.state_vec[:500] * (np.abs(self.fitresult_gauss[1])
                                                                         + self.fitresult_gauss[4])
                                                  + self.fitresult_gauss[1]), color='red', label='state assignment')
        ax_state_assign.set_xlabel('timestep')
        ax_state_assign.set_ylabel('I (arb. un.)')
        ax_state_assign.legend()


