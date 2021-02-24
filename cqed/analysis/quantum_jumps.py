# Developed and written by the cQED team of the Kouwenhoven lab at QuTech 2020-2021

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from hmmlearn import hmm
import cqed.utils.datahandling as dh
import cqed.utils.fit_functions as fitf
from lmfit import minimize, Parameters


class QntmJumpTrace:
    """ The purpose of this class is to analyze time traces of transitions between two states. Three different
    techniques to extract the transition rates between the two states are available via the latching_pipeline,
    hmm_pipeline, and psd_pipeline methods.
    The input data is assumed to be points in a 2d plane, e.g. IQ points commonly encountered in dispersive readout.
    """

    def __init__(self, data, dt, data_complex=True, theta=None, n_integration=1):
        """
        Class to analyse quantum jumps traces and extract transition
        rates between the identified states

        @param data: either a complex 2d-array, or a tuple of 2d arrays (real, imag). If 2d array it is assumed that it
        is less time traces than data points per trace.
        @param dt: time between two data points in seconds
        @param data_complex: bool, default=True. If False data is expected to be a tuple of 2d arrays
        @param theta: angle to rotate the raw input data in the complex plane. If not specified the data is rotated for
        max variance in the real axis.
        @param n_integration: Number of samples to average together to decrease noise at the expense of decreased time
        resolution. Note that this kwarg has no influence on the PSD in the psd pipeline, as for that analysis the raw
        input data before integration is used.
        """

        if data_complex:
            self.raw_data = np.array(data)
        else:
            self.raw_data = np.array(data[0]) + 1.j * np.array(data[1])

        # take care of 1d and 2d arrays as input and handle them on equal footing
        self.dat_dims = self.raw_data.shape
        if len(self.dat_dims) < 2:
            self.raw_data = np.expand_dims(self.raw_data, axis=0)
            self.dat_dims = (1, self.dat_dims[0])
        else:
            if self.dat_dims[0] > self.dat_dims[1]:
                self.raw_data = self.raw_data.T
                self.dat_dims = (self.dat_dims[1], self.dat_dims[0])

        # Integrating data
        if n_integration > 1:
            divisors = int(self.raw_data[0, :, ].size // n_integration)
            self.integrated_data = np.zeros((self.dat_dims[0], divisors), dtype=np.complex64)
            for ii in range(self.dat_dims[0]):
                self.integrated_data[ii, :] = np.mean(
                    self.raw_data[ii, 0:divisors * n_integration].reshape(-1, n_integration), axis=1)

        else:
            self.integrated_data = self.raw_data

        self.dt = float(dt * n_integration)
        self.n_integration = n_integration

        # rotate the data for max variance in the real axis
        if theta is None:
            thetas = np.zeros(self.dat_dims[0])
            for i in range(self.dat_dims[0]):
                thetas[i] = dh.max_variance_angle(self.integrated_data[i])
            self.theta = np.mean(thetas)
        else:
            self.theta = theta

        self.raw_data_rot = np.zeros_like(self.raw_data, dtype=complex)
        self.integrated_data_rot = np.zeros_like(self.integrated_data, dtype=complex)

        for i in range(self.dat_dims[0]):
            self.raw_data_rot[i] = dh.rotate_data(self.raw_data[i], self.theta)
            self.integrated_data_rot[i] = dh.rotate_data(self.integrated_data[i], self.theta)

        self.SNR = None

        # attributes for latching filter pipeline
        self.raw_hist = []
        self.state_vec = np.empty_like(self.integrated_data)
        self.dwell_l = []
        self.dwell_h = []
        self.hist_dwell_l = np.empty((self.dat_dims[0], 2), dtype=object)
        self.hist_dwell_h = np.empty((self.dat_dims[0], 2), dtype=object)
        self.rate_lh = np.zeros((self.dat_dims[0], 2), dtype=float)
        self.rate_hl = np.zeros((self.dat_dims[0], 2), dtype=float)
        self.n_sigma_filter = 1.
        self.fitresult_gauss = None
        self._dwellhist_guess = [-0.01, 1000.]  # some hardcoded guesses for the linear fit of dwell time histogram
        self._filter_params = None
        self.rate_lh_err = 0
        self.rate_hl_err = 0

        # attributes for PSD pipeline
        self.ys_fft = np.zeros(self.dat_dims[1])
        self.fs = None
        self.rate_lh_psd = None
        self.rate_hl_psd = None
        self.fit_accuracy_psd = None
        self.fitresult_lorentzian = None
        self.min_data = None
        self.max_data = None
        self.rate_lh_psd_err = None
        self.rate_hl_psd_err = None
        self._err_R = 0.
        self._psd_m = 0
        self._err_Gamma = 0.

        # attributes for the hidden markov pipeline
        self.hmm_model = hmm.GaussianHMM(n_components=2)
        self.state_vec_hmm = []
        self.rate_hl_hmm = None
        self.rate_lh_hmm = None

    def hmm_pipeline(self, n_iter=100, n_bins=100, dbl_gauss_p0=None):
        """
        Use the hidden markov model analysis provided by the hmmlearn python package to extract transition rates between
        two states.
        @param n_iter: Number of iterations for the fit routine of the hmmlearn package. Default is 100.
        @return: None
        """
      
        self.hmm_model.n_iter = n_iter
        self.hmm_model.init_params = "t"  # tell the model which parameters are not initialized
        self.hmm_model.params="cmts"  # allow model to update all parameters
        
        flattened_input = self.integrated_data_rot.real.reshape(-1, 1)
        seq_len = (np.ones(self.dat_dims[0], dtype=int) * self.dat_dims[1] / self.n_integration).astype(int)

        if self.fitresult_gauss is None:
            self._double_gauss_routine(n_bins, dbl_gauss_p0)
        
        # estimate the model parameters to improve convergence
        # the relative height of the peaks gives us an idea of the start probability
        self.hmm_model.startprob_ = np.array([self.fitresult_gauss.params["c1"],self.fitresult_gauss.params["c2"]]) \
                                    / (self.fitresult_gauss.params["c1"] + self.fitresult_gauss.params["c2"])

        self.hmm_model.means_ = np.array([[self.fitresult_gauss.params["mu1"]], [self.fitresult_gauss.params["mu2"]]])
        self.hmm_model.covars_ = np.sqrt(np.array([[self.fitresult_gauss.params["sig1"]],
                                                   [self.fitresult_gauss.params["sig2"]]]))
        
        self.hmm_model.fit(flattened_input, lengths=seq_len)
        self.state_vec_hmm = self.hmm_model.predict(flattened_input, lengths=seq_len).reshape(self.dat_dims)

        # calculate the eigenvalues
        a = np.linalg.eig(np.array(self.hmm_model.transmat_))
        # calculate some characteristic 1/rate TODO: check if this is correct!
        tau = -self.dt/np.log(np.abs(np.min(a[0])))
        # get the stationary distribution to calculate two times from one
        stat = self.hmm_model.get_stationary_distribution()

        self.rate_hl_hmm = stat[1]/(stat[1]+stat[0]) / tau
        self.rate_lh_hmm = stat[0]/(stat[0]+stat[1]) / tau

    @staticmethod
    def _create_hist(data, n_bins, rng):
        """Calculate a histogram and return an x-axis that is of same length. The numbers of the x-axis represent
         the middle of the corresponding bins
         @param data: array to histogram
         @param n_bins: number of bins for the histogramming
         @param rng: range passed to np.histogram
         @return: tuple of: middle of bins, number of counts"""

        n, bins = np.histogram(data, bins=n_bins, density=False, range=rng)
        x_axis = bins[:-1] + (bins[1] - bins[0]) / 2.

        return x_axis, n

    @staticmethod
    def _fit_dbl_gaussian(xdata, ydata, min_data=-np.inf, max_data=np.inf, start_param_guess=None):
        """ Fit a double gaussian distribution using lmfit's minimize method.
        @param xdata: (array)
        @param ydata: (array)
        @param min_data: lower bound for means of double exponential fit
        @param max_data: upper bound for means of double exponential fit
        @param start_param_guess: starting parameters for the fitting algorithm, expects iterable of length 6 with
            [amplitude1, mean1, standard deviation1, amplitude2, mean2, standard deviation1], if None
            cqed.utils.fit_functions.dbl_gaussian_guess_means and gaussian_guess_sigma_A are used
        @return: out (output of the lmfit.minimize method)
        """

        if start_param_guess is None:
            mu1, mu2 = fitf.dbl_gaussian_guess_means(xdata, ydata, threshold=0.1) 
            sig, c = fitf.gaussian_guess_sigma_A(xdata, ydata, threshold=0.2)             
            params = Parameters()
            params.add('c1', value=c, min=0.0, max=5*c)
            params.add('c2', value=c, min=0.0, max=5*c)
            params.add('mu1', value=mu1, min=min_data, max=max_data)
            params.add('mu2', value=mu2, min=min_data, max=max_data)
            params.add('sig1', value=sig, min=0.05*sig, max=20*sig)
            params.add('sig2', value=sig, min=0.05*sig, max=20*sig)
        else:
            params = Parameters()
            params.add('c1', value=start_param_guess[0], min=0.0, max=np.inf)
            params.add('c2', value=start_param_guess[3], min=0.0, max=np.inf)
            params.add('mu1', value=start_param_guess[1], min=min_data, max=max_data)
            params.add('mu2', value=start_param_guess[4], min=min_data, max=max_data)
            params.add('sig1', value=start_param_guess[2], min=0.0, max=20*start_param_guess[2])
            params.add('sig2', value=start_param_guess[5], min=0.0, max=20*start_param_guess[5])
        
        out = minimize(fitf.residual, params, args=(xdata, ydata, fitf.dbl_gaussian))
        # fit, conv = curve_fit(fitf.dbl_gaussian, xdata, ydata, p0=start_param_guess, **kwargs)

        # ensure that the gaussian with smaller mean is the first set of data
        # if fit[4] < fit[1]:
        #     fit = np.array([*fit[3:], *fit[:3]])

        return out  # fit, conv

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

    def _double_gauss_routine(self, n_bins, dbl_gauss_p0):
        """
        Histogram the real part of the rotated data all at once, i.e. flatten the input array in case it's 2d
        then fit a double gaussian distribution.
        @param n_bins: number of bins used for the histogram
        @param dbl_gauss_p0: start parameters for the double gaussian fit routine. Expects iterable of size 6:
        (amplitude1, mean1, sigma1, amplitude2, mean2, sigma2)
        @return None: just fills the attributes fitresult_gauss and SNR
        """
        self.min_data = np.min(self.integrated_data_rot.real.flatten())
        self.max_data = np.max(self.integrated_data_rot.real.flatten())
        range_data = self.max_data - self.min_data
        x_max = self.max_data + 10*range_data
        x_min = self.min_data - 10*range_data
        n_bins = int(n_bins * (x_max-x_min)/(self.max_data-self.min_data))
        self.raw_hist = self._create_hist(self.integrated_data_rot.real.flatten(), n_bins=n_bins, rng=(x_min, x_max))

        # TODO: do we want these bounds? I encountered that doing this it sometimes gives errors when
        # trying to fit two Gaussians to data that is only one Gaussian, because it just finds mu2 very
        # far away such that only it's tail overlaps the data and it ends up resulting in unrealistic results
        if dbl_gauss_p0 is not None:
            self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_hist[0], self.raw_hist[1],
                                                          start_param_guess=dbl_gauss_p0, min_data=self.min_data,
                                                          max_data=self.max_data)
        else:
            self.fitresult_gauss = self._fit_dbl_gaussian(self.raw_hist[0], self.raw_hist[1], min_data=self.min_data,
                                                          max_data=self.max_data)
            average_sigma = 0.5 * (self.fitresult_gauss.params["sig1"] + self.fitresult_gauss.params["sig2"])
            self.SNR = np.abs((self.fitresult_gauss.params["mu2"] - self.fitresult_gauss.params["mu1"]) / average_sigma)

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
        @param state_filter_prms: (tuple) Parameters used for the state assignment algorithm if override_gaussfit=True.
        Expects input of the form ((mean1, mean2), (standard deviation1, standard deviation2))
        @param n_sigma_filter: (float) prefactor to adjust the thresholding for the state assignment algorithm.
        @return: class instance itself
        """

        self.n_sigma_filter = n_sigma_filter

        if self.fitresult_gauss is None:
            self._double_gauss_routine(n_bins, dbl_gauss_p0)

        for i in range(self.dat_dims[0]):
            if not override_gaussfit:
                sg1 = self.n_sigma_filter * self.fitresult_gauss.params["sig1"].value
                sg2 = self.n_sigma_filter * self.fitresult_gauss.params["sig2"].value
                if self.fitresult_gauss.params["mu2"].value > self.fitresult_gauss.params["mu1"].value:
                    self.state_vec[i], _dwell_l, _dwell_h = self._latching_filter(self.integrated_data_rot[i].real,
                                                                                  (self.fitresult_gauss.params["mu1"].value,
                                                                                   self.fitresult_gauss.params["mu2"].value),
                                                                                  (sg1, sg2))
                else:
                    self.state_vec[i], _dwell_l, _dwell_h = self._latching_filter(self.integrated_data_rot[i].real,
                                                                                  (self.fitresult_gauss.params["mu2"].value,
                                                                                   self.fitresult_gauss.params["mu1"].value),
                                                                                  (sg2, sg1))
                self.dwell_l = np.concatenate((self.dwell_l, _dwell_l))
                self.dwell_h = np.concatenate((self.dwell_h, _dwell_h))

            else:
                self._filter_params = state_filter_prms

                self.state_vec[i], _dwell_l, _dwell_h = self._latching_filter(self.integrated_data_rot[i].real,
                                                                              self._filter_params[0],
                                                                              n_sigma_filter
                                                                              * self._filter_params[1])
                self.dwell_l = np.concatenate((self.dwell_l, _dwell_l))
                self.dwell_h = np.concatenate((self.dwell_h, _dwell_h))

        bins_edges = np.arange(1, np.max(self.dwell_l)+2., 1.) - 0.5

        self.hist_dwell_l = self._create_hist(self.dwell_l, bins_edges, (np.min(self.dwell_l), np.max(self.dwell_l)))

        bins_edges = np.arange(1, np.max(self.dwell_h)+2., 1.) - 0.5
        self.hist_dwell_h = self._create_hist(self.dwell_h, bins_edges, (np.min(self.dwell_h), np.max(self.dwell_h)))

        # since the histogram has zeros as entries, filter those before passing to curve_fit
        log_yl = np.log(self.hist_dwell_l[1])
        inds = np.where(np.isfinite(log_yl))[0]
        x_l = self.hist_dwell_l[0][inds]

        self.rate_lh, conv = curve_fit(fitf.lin_func, x_l, log_yl[inds], p0=[self._dwellhist_guess[0],
                                                                             self._dwellhist_guess[1]])
        self.rate_lh_err = np.sqrt(np.diag(conv)) / self.dt

        # since the histogram has zeros as entries, filter those before passing to curve_fit
        log_yh = np.log(self.hist_dwell_h[1])
        inds = np.where(np.isfinite(log_yh))[0]
        x_h = self.hist_dwell_h[0][inds]

        self.rate_hl, conv = curve_fit(fitf.lin_func, x_h, log_yh[inds], p0=[self._dwellhist_guess[0],
                                                                             self._dwellhist_guess[1]])
        self.rate_hl_err = np.sqrt(np.diag(conv)) / self.dt

        # rescale the rates to units of Hz
        self.rate_lh[0] = np.abs(self.rate_lh[0] / self.dt)
        self.rate_hl[0] = np.abs(self.rate_hl[0] / self.dt)

        return self

    def plot_latching_analysis(self, figsize=(16, 9)):
        """
        Function to plot an overview of the latching_pipeline fit results.
        @param figsize: passed to matplotlib.pyplot.figure to adjust the figure size
        @return: matplotlib figure with multiple axes showing the analysis results.
        """

        fig = plt.figure(1, figsize=figsize)
        fig.clf()
        dim_ax_raw = [0.05, 0.7, 0.2, 0.25]
        dim_ax_rot = [0.05, 0.375, 0.2, 0.25]
        dim_ax_dwell_hist = [0.05, 0.05, 0.2, 0.25]
        dim_ax_trace = [0.3, 0.55, 0.59, 0.4]
        dim_ax_histy = [0.9, 0.55, 0.09, 0.4]
        dim_ax_state_assign = [0.3, 0.05, 0.69, 0.4]

        ax_raw = fig.add_axes(dim_ax_raw)
        ax_rot = fig.add_axes(dim_ax_rot)
        ax_dwell_hist = fig.add_axes(dim_ax_dwell_hist)
        ax_trace = fig.add_axes(dim_ax_trace)
        ax_histy = fig.add_axes(dim_ax_histy)
        ax_state_assign = fig.add_axes(dim_ax_state_assign)

        ax_raw.hist2d(self.raw_data[0].real, self.raw_data[0].imag, bins=50)
        ax_raw.axis('equal')
        ax_raw.set_title('raw data')
        ax_raw.set_xlabel('I (arb. un.)')
        ax_raw.set_ylabel('Q (arb. un.)')

        ax_rot.hist2d(self.raw_data_rot[0].real, self.raw_data_rot[0].imag, bins=50)
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
                ax_dwell_hist.plot(self.hist_dwell_l[0], fitf.exp_func(self.hist_dwell_l[0], -self.rate_lh[0],
                                                                        np.exp(self.rate_lh[1])), 'k')
                ax_dwell_hist.plot(self.hist_dwell_h[0], fitf.exp_func(self.hist_dwell_h[0], -self.rate_hl[0],
                                                                        np.exp(self.rate_hl[1])), 'k')

        ax_dwell_hist.set_yscale('log')
        ax_dwell_hist.set_title('histogram of dwell times')
        ax_dwell_hist.set_ylabel('norm. occurence')
        ax_dwell_hist.set_xlabel('dwell time')
        ax_dwell_hist.legend()

        if self._filter_params is None:
            ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss.params['mu1'] - self.n_sigma_filter
                                  * self.fitresult_gauss.params['sig1'],
                                  self.fitresult_gauss.params['mu1'] + self.n_sigma_filter
                                  * self.fitresult_gauss.params['sig1'], alpha=0.4, color='tab:orange')
            ax_trace.axhline(self.fitresult_gauss.params['mu1'].value, ls='dashed', color='tab:orange')

            ax_trace.fill_between(np.arange(0, 400, 1), self.fitresult_gauss.params['mu2'] - self.n_sigma_filter
                                  * self.fitresult_gauss.params['sig2'],
                                  self.fitresult_gauss.params['mu2'] + self.n_sigma_filter
                                  * self.fitresult_gauss.params['sig2'], alpha=0.4, color='tab:green')
            ax_trace.axhline(self.fitresult_gauss.params['mu2'].value, ls='dashed', color='tab:green')

            ax_trace.plot(np.arange(0, 400, 1), self.raw_data_rot[0].real[:400], 'k.-', label='real rotated data')
            ax_trace.set_xlim(-1, 401)

            if self.fitresult_gauss.params['mu1'] < self.fitresult_gauss.params['mu2']:
                ax_trace.set_ylim(self.fitresult_gauss.params['mu1'] - 3 * self.fitresult_gauss.params['sig1'],
                                  self.fitresult_gauss.params['mu2'] + 3 * self.fitresult_gauss.params['sig2'])
            else:
                ax_trace.set_ylim(self.fitresult_gauss.params['mu2'] - 3 * self.fitresult_gauss.params['sig2'],
                                  self.fitresult_gauss.params['mu1'] + 3 * self.fitresult_gauss.params['sig1'])
        else:
            ax_trace.fill_between(np.arange(0, 400, 1), self._filter_params[0][0] - self.n_sigma_filter *
                                  self._filter_params[1][0], self._filter_params[0][0] + self.n_sigma_filter *
                                  self._filter_params[1][0], alpha=0.4, color='tab:orange')
            ax_trace.axhline(self._filter_params[0][0], ls='dashed', color='tab:orange')

            ax_trace.fill_between(np.arange(0, 400, 1), self._filter_params[0][1] - self.n_sigma_filter *
                                  self._filter_params[1][1], self._filter_params[0][1] + self.n_sigma_filter *
                                  self._filter_params[1][1], alpha=0.4, color='tab:green')
            ax_trace.axhline(self._filter_params[0][1], ls='dashed', color='tab:green')

            ax_trace.plot(np.arange(0, 400, 1), self.raw_data_rot[0].real[:400], 'k.-', label='real rotated data')
            ax_trace.set_xlim(-1, 401)

            ax_trace.set_ylim(self._filter_params[0][0] - 3 * self._filter_params[1][0],
                              self._filter_params[0][1] + 3 * self._filter_params[1][1])

        ax_trace.set_title('rotated data trace excerpt')
        ax_trace.set_xlabel('timestep')
        ax_trace.set_ylabel('I (arb. un.)')
        ax_histy.barh(self.raw_hist[0], self.raw_hist[1], (self.raw_hist[0][1] - self.raw_hist[0][0]) * 0.8,
                      label='rot. data')
        ax_histy.plot(fitf.dbl_gaussian(self.raw_hist[0], self.fitresult_gauss.params), self.raw_hist[0],
                      'k', label='fit')
        ax_histy.tick_params(direction='in', labelleft=False)
        ax_histy.set_ylim(ax_trace.get_ylim())
        ax_histy.set_xlabel('norm. occurence')
        ax_histy.set_title('histogram rot. data')
        ax_histy.legend()

        ax_state_assign.plot(np.arange(0, 500, 1), self.raw_data_rot.real[0][:500], 'k.-', label='real rotated data')

        if self._filter_params is None:
            ax_state_assign.plot(np.arange(0, 500, 1), (self.state_vec[0][:500]
                                                        * (np.abs(self.fitresult_gauss.params['mu1']) +
                                                           self.fitresult_gauss.params['mu2'])
                                                        + self.fitresult_gauss.params['mu1']), color='red',
                                 label='state assignment')
        else:
            ax_state_assign.plot(np.arange(0, 500, 1), (self.state_vec[0][:500]
                                                        * (np.abs(self._filter_params[0][0]) +
                                                           self._filter_params[0][1])
                                                        + self._filter_params[0][0]), color='red',
                                 label='state assignment')

            ax_state_assign.text(10, self._filter_params[0][1]+self._filter_params[1][1], 'Double Gauss fit overridden',
                                 color='red')

        ax_state_assign.set_xlabel('timestep')
        ax_state_assign.set_ylabel('I (arb. un.)')
        ax_state_assign.legend()
        ax_state_assign.set_ylim(ax_trace.get_ylim())

        return fig

    @staticmethod
    def _calculate_PSD(ys):
        """
        returns the PSD of a timeseries and freuency axis.
        @param ys: time series to analyze
        @return fs: frequency axis
        @return ys_fft: result of Fourier analysis
        """
        ys_fft = np.abs(np.fft.fft(ys)) ** 2

        fmax = 1.
        df = 1. / len(ys)
        pts_fft = int(len(ys) / 2)
        fs1 = np.linspace(0, fmax / 2. - df, pts_fft)
        fs2 = np.linspace(fmax / 2., df, pts_fft)
        fs = np.append(fs1, -fs2)
        return fs, ys_fft

    def psd_pipeline(self, n_bins=100, dbl_gauss_p0=None, m_guess=0.8e-1, gamma_guess=1e-4):
        """
        Perform the analysis of time traces using a fit of double gaussian and a fit of a Lorentzian to the PSD
        to extract transition rates.
        @param n_bins: Number of bins used for the double gaussian histogram
        @param dbl_gauss_p0: Starting parameters for the double gaussian fit of the form [amplitude1, mean1, sigma1,
        amplitude2, mean2, sigma2]
        @param m_guess: upper bound for the data used for fitting the lorentzian to the PSD
        @param gamma_guess: starting parameter for the fit of the lorentzian to the PSD
        """

        if self.fitresult_gauss is None:
            self._double_gauss_routine(n_bins, dbl_gauss_p0)

        # calculating the error of R based on the fit uncertainties of c1, c2
        R = self.fitresult_gauss.params["c2"].value / self.fitresult_gauss.params["c1"].value
        self._err_R = np.sqrt((-self.fitresult_gauss.params["c2"].value / self.fitresult_gauss.params["c1"].value**2
                               * self.fitresult_gauss.params["c1"].stderr)**2
                              + (1. / self.fitresult_gauss.params["c1"].value
                                 * self.fitresult_gauss.params["c2"].stderr)**2
                              )

        # Calculating and plotting PSD
        for i in range(self.dat_dims[0]):
            fs, ys_fft_i = self._calculate_PSD(self.raw_data_rot[i])
            self.ys_fft += ys_fft_i/self.dat_dims[0]
        self.fs = fs

        # Plotting initial guess for the fit
        m = np.argmin(np.abs(fs - m_guess))  # TODO: This is semi-hard coded
        xdat = np.real(self.fs[1:m])
        ydat = self.ys_fft[1:m]
        self._psd_m = m  # save it just for plotting later

        # Fitting Lorentzian
        params = Parameters()
        params.add('Gamma', value=gamma_guess)
        params.add('a', value=gamma_guess * np.mean(ydat[0:9]))
        params.add('b', value=np.mean(ydat[-10:-1]))

        out = minimize(fitf.residual, params, args=(xdat, ydat, fitf.QPP_Lorentzian))
        self.fitresult_lorentzian = out

        Gamma = out.params["Gamma"].value
        self._err_Gamma = out.params["Gamma"].stderr

        Gamma1 = 2. * Gamma / (1. + R)
        Gamma1_err = np.sqrt((2. / (1. + R) * self._err_Gamma)**2 + (-2. * Gamma / (1. + R)**2 * self._err_R)**2)

        Gamma2 = 2. * R * Gamma / (1. + R)
        Gamma2_err = np.sqrt((2. * R / (1. + R) * self._err_Gamma)**2
                             + ((2. * Gamma / (1. + R) - 2. * Gamma * R / (1. + R)**2) * self._err_R)**2)

        # multiply the rates with the integration number, because the PSD and its rates is calculated from the raw,
        # unintegrated data
        if self.fitresult_gauss.params["mu1"].value > self.fitresult_gauss.params["mu2"].value:
            self.rate_lh_psd = Gamma2 / self.dt * self.n_integration
            self.rate_lh_psd_err = Gamma2_err / self.dt * self.n_integration

            self.rate_hl_psd = Gamma1 / self.dt * self.n_integration
            self.rate_hl_psd_err = Gamma1_err / self.dt * self.n_integration
        else:
            self.rate_lh_psd = Gamma1 / self.dt * self.n_integration
            self.rate_lh_psd_err = Gamma1_err / self.dt * self.n_integration

            self.rate_hl_psd = Gamma2 / self.dt * self.n_integration
            self.rate_hl_psd_err = Gamma2_err / self.dt * self.n_integration

        # Calculate measure of fit accuracy
        self.fit_accuracy_psd = 1. / (self.fitresult_gauss.redchi * self.fitresult_lorentzian.redchi)

    def plot_psd_analysis(self, figsize=(12, 4)):
        """ Plot the PSD analysis, i.e. the double Gaussian distribution with the fit overlaid, and the PSD with the
        Lorentzian fit.
        @param figsize: tuple passed to matplotlib.pyplot.subplots to adjust the size of the figure.
        @return None
        """

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(self.raw_hist[0], self.raw_hist[1], 'bo')
        axes[0].plot(self.raw_hist[0], fitf.dbl_gaussian(self.raw_hist[0], self.fitresult_gauss.params), 'r')
        axes[0].axvline(self.fitresult_gauss.params["mu1"].value, color='r')
        axes[0].axvline(self.fitresult_gauss.params["mu2"].value, color='r')
        if (self.min_data is not None) and (self.max_data is not None):
            axes[0].set_xlim([self.min_data, self.max_data])
        axes[0].set_xlabel('I (arb. un.)')
        axes[0].set_ylabel('Counts')

        inds = np.where(self.fs >= 0)[0]
        axes[1].plot(self.fs[inds], self.ys_fft[inds], 'b-', label='data')
        axes[1].plot(self.fs[1:self._psd_m], fitf.QPP_Lorentzian(self.fs[1:self._psd_m],
                                                                 self.fitresult_lorentzian.params), 'r-', label='fit')
        axes[1].axvline(self.fs[self._psd_m], ls='dashed', color='black', label='cutoff for fit')
        axes[1].axvline(self.fs[1], ls='dashed', color='black')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_xscale('log')
        axes[1].legend()
