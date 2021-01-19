import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import cqed.utils.datahandling as dh
import cqed.utils.fit_functions as fitf


class QntmJumpTrace:

    def __init__(self, data, data_complex=True):
        """
        Class to analyse quantum jumps traces and extract transition
        rates between the identified states

        @param data: either a complex 1d-array, or a tuple (real, imag)
        @param data_complex: bool, default=True
        """

        if data_complex:
            self.raw_data = data
        else:
            self.raw_data = data[0] + 1.j * data[1]

        self.n_sigma_filter = 1.
        self.raw_data_rot = np.empty_like(self.raw_data)
        self.theta = 0.
        self.raw_hist = []
        self.state_vec = np.empty_like(self.raw_data)
        self.dwell_l = []
        self.dwell_h = []
        self.hist_dwell_l = []
        self.hist_dwell_h = []
        self.rate_lh = None
        self.rate_hl = None

        self.fitresult_gauss = None

    @staticmethod
    def _create_hist(data, n_bins):
        """Calculate a histogram and return an x-axis that is of same length. The numbers of the x-axis represent
         the middle of the corresponding bins
         @param data: array to histogram
         @param n_bins: number of bins for the histogramming
         @return: tuple of: middle of bins, normalized occurrence"""

        n, bins = np.histogram(data, bins=n_bins, density=True)
        x_axis = bins[:-1] + (bins[1] - bins[0]) / 2.

        return x_axis, n

    @staticmethod
    def _fit_dbl_gaussian(data, n_bins=100, start_param_guess=None, **kwargs):
        """ Histogram an input data set, then fit a double gaussian distribution using scipy.curve_fit.
        @param data: raw data set to fit a double gaussian distribution to
        @param n_bins: number of bins to be used for the histogramming
        @param start_param_guess: starting parameters for the fitting algorithm, expects iterable of length 6 with
            [amplitude1, mean1, standard deviation1, amplitude2, mean2, standard deviation1]
        @return: array of the fit parameters
        """

        I_ax, n = QntmJumpTrace._create_hist(data, n_bins)

        if start_param_guess is None:
            start_param_guess = [n[20], I_ax[10], I_ax[10] - I_ax[0], n[-20], I_ax[-10], I_ax[10] - I_ax[0]]

        fit, conv = curve_fit(fitf.dbl_gaussian, I_ax, n, p0=start_param_guess, **kwargs)

        # ensure that the negative gaussian is the first set of data
        if fit[4] < fit[1]:
            fit = np.array([*fit[3:], *fit[:3]])

        return fit

    @staticmethod
    def _latching_filter(arr, means, sigm):
        """ Assign states of a telegram signal with noise. For the algorithm to register a jump the point has to
         exceed the threshold given by mean +- sigma. Otherwise the point will be assigned the same state as the
         previous one
         @param arr: time series of points
         @param means: tuple (mean of emissions of state 1, mean of emission of state 2)
         @param sigm: tuple (value below/above the mean where a jump is assigned for state 1, value for state 2)
         @return state, dwell_l, dwell_h: three arrays. state holds the assigned state (0 for the states with mean1,
          1 for the state with mean2), dwell_l array of dwell times in the state centered around mean1,
          dwell_h array of dwell times in the state centered around mean2.
          """

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

    def latching_pipeline(self, n_bins=100, dbl_gauss_p0=None, override_gaussfit=False, state_filter_prms=None,
                          n_sigma_filter=1.):
        """
        Perform the entire data analysis from raw IQ trace to transition rates between two states
        Usage: after initializing a class instance run this function to perform analysis comprising of:
        rotating the data for max variance in real part, histogram and fit a double gaussian dist. to the real part
        assign states and find dwell times using the latching_filter. Histogram the dwell times and fit an exponential
        distribution to find the transition rates.
        @param n_bins: number of bins used for the histogram of the real part of the rotated data
        @param dbl_gauss_p0: starting parameters for the double gaussian fit
        @param override_gaussfit: (bool) default False: use the fit parameters found using the internal dbl gauss fit
        route. True: use the means and sigmas provided through the state_filter_prms kwarg.
        @param state_filter_prms: (tuple) Parameters used for the state assignment algorithm is override_gaussfit=True.
        Expects input of the form ((mean1, mean2), (standard deviation1, standard deviation2))
        @param n_sigma_filter: (float) prefactor to adjust the thresholding for the state assignment algorithm.
        @return: class instance itself
        """

        self.n_sigma_filter = n_sigma_filter
        self.theta = dh.max_variance_angle(self.raw_data)
        self.raw_data_rot = dh.rotate_data(self.raw_data, self.theta)

        if dbl_gauss_p0 is not None:
            self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_data_rot.real, start_param_guess=dbl_gauss_p0)
        else:
            try:
                self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_data_rot.real)
            except RuntimeError:
                self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_data_rot.real, maxfev=int(1e4))

        self.raw_hist = self._create_hist(self.raw_data_rot.real, n_bins)

        if not override_gaussfit:
            self.state_vec, self.dwell_l, self.dwell_h = self._latching_filter(self.raw_data_rot.real,
                                                                               self.fitresult_gauss[[1, 4]],
                                                                               n_sigma_filter *
                                                                               self.fitresult_gauss[[2, 5]])
        else:
            self.fitresult_gauss = np.array([1., state_filter_prms[0][0], state_filter_prms[1][0],
                                             1., state_filter_prms[0][1], state_filter_prms[1][1]])

            self.state_vec, self.dwell_l, self.dwell_h = self._latching_filter(self.raw_data_rot.real,
                                                                               self.fitresult_gauss[[1, 4]],
                                                                               n_sigma_filter *
                                                                               self.fitresult_gauss[[2, 5]])

        try:
            self.hist_dwell_l = self._create_hist(self.dwell_l, np.max(self.dwell_l))
            self.hist_dwell_h = self._create_hist(self.dwell_h, np.max(self.dwell_h))
        except ValueError:
            self.hist_dwell_l = None
            self.hist_dwell_h = None

        if self.hist_dwell_h is not None:
            try:
                # since the histogram has zeros as entries, filter those before passing to curve_fit
                log_yl = np.log(self.hist_dwell_l[1])
                inds = np.where(np.isfinite(log_yl))[0]
                x_l = self.hist_dwell_l[0][inds]
                y_l = log_yl[inds]

                self.rate_lh = curve_fit(fitf.lin_func, x_l, y_l, p0=[-0.01, 1.])

                # since the histogram has zeros as entries, filter those before passing to curve_fit
                log_yh = np.log(self.hist_dwell_h[1])
                inds = np.where(np.isfinite(log_yh))[0]
                x_h = self.hist_dwell_h[0][inds]
                y_h = log_yh[inds]

                self.rate_hl = curve_fit(fitf.lin_func, x_h, y_h, p0=[-0.01, 1.])

            except ValueError:
                self.rate_lh = None
                self.rate_hl = None
        else:
            self.rate_lh = None
            self.rate_hl = None

        return self

    def plot_analysis(self, figsize=(16, 9)):
        """
        Function to plot an overview of the latching_pipeline fit results.
        @param figsize: passed to matplotlib.pyplot.figure to adjust the figure size
        @return: matplotlib figure with multiple axes showing the analysis results.
        """

        plt.figure(1, figsize=figsize)
        plt.gcf()
        dim_ax_raw = [0.05, 0.7, 0.2, 0.25]
        dim_ax_rot = [0.05, 0.375, 0.2, 0.25]
        dim_ax_dwell_hist = [0.05, 0.05, 0.2, 0.25]
        dim_ax_trace = [0.3, 0.55, 0.59, 0.4]
        dim_ax_histy = [0.9, 0.55, 0.09, 0.4]
        dim_ax_state_assign = [0.3, 0.05, 0.69, 0.4]

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

        if self.hist_dwell_h is None:
            pass
        else:
            ax_dwell_hist.plot(self.hist_dwell_l[0], self.hist_dwell_l[1], 'o', mfc='none', label='low state')
            ax_dwell_hist.plot(self.hist_dwell_h[0], self.hist_dwell_h[1], 'o', mfc='none', label='higher state')

            if self.rate_lh is not None:
                ax_dwell_hist.plot(self.hist_dwell_l[0], fitf.exp_func(self.hist_dwell_l[0], self.rate_lh[0][0],
                                                                        np.exp(self.rate_lh[0][1])), 'k')
                ax_dwell_hist.plot(self.hist_dwell_h[0], fitf.exp_func(self.hist_dwell_h[0], self.rate_hl[0][0],
                                                                        np.exp(self.rate_hl[0][1])), 'k')

            ax_dwell_hist.set_xlim(1, np.max([np.max(self.hist_dwell_l[0]), np.max(self.hist_dwell_h[0])]))

        ax_dwell_hist.set_yscale('log')
        ax_dwell_hist.set_title('histogram of dwell times')
        ax_dwell_hist.set_ylabel('norm. occurence')
        ax_dwell_hist.set_xlabel('dwell time')
        ax_dwell_hist.legend()

        ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss[1] - self.n_sigma_filter * self.fitresult_gauss[2],
                              self.fitresult_gauss[1] + self.n_sigma_filter * self.fitresult_gauss[2], alpha=0.4, color='tab:orange')
        ax_trace.axhline(self.fitresult_gauss[1], ls='dashed', color='tab:orange')
        ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss[4] - self.n_sigma_filter * self.fitresult_gauss[5],
                              self.fitresult_gauss[4] + self.n_sigma_filter * self.fitresult_gauss[5], alpha=0.4, color='tab:green')
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
        ax_histy.plot(fitf.dbl_gaussian(self.raw_hist[0], *self.fitresult_gauss), self.raw_hist[0], 'k', label='fit')
        ax_histy.tick_params(direction='in', labelleft=False)
        ax_histy.set_ylim(ax_trace.get_ylim())
        ax_histy.set_xlabel('norm. occurence')
        ax_histy.set_title('histogram rot. data')
        ax_histy.legend()

        ax_state_assign.plot(np.arange(0, 500, 1), self.raw_data_rot.real[:500], 'k.-', label='real rotated data')
        ax_state_assign.plot(np.arange(0, 500, 1), (self.state_vec[:500] * (np.abs(self.fitresult_gauss[1])
                                                                            + self.fitresult_gauss[4])
                                                    + self.fitresult_gauss[1]), color='red', label='state assignment')
        ax_state_assign.set_xlabel('timestep')
        ax_state_assign.set_ylabel('I (arb. un.)')
        ax_state_assign.legend()
