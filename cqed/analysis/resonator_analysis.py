from xarray import DataArray, merge
from resonator_tools.circuit import notch_port
import numpy as np
import matplotlib.pyplot as plt


def fit_resonator(array, fit_axis, plot_fit=False):
    """
    Takes an xarray with data variables called 'amplitude' and 'phase' and returns an xarray consisting of the original
    raw data and the resonator fit parameters as a function of the coordinate 'fit_axis' (which is one of the
    coordinates of the input xarray). Fitting is performed using the 'notch_port' class of the circle-fit routine
    provided by https://github.com/sebastianprobst/resonator_tools
    @param array: xarray with the data variables amplitude (linear units) and phase (radians), has to have the
        coordinate frequency, and at least one further coordinate
    @param fit_axis: coordinate of the xarray along which the fits should be performed
    @param plot_fit: If True shows all raw data with fits overlay
    @return: xarray consisting of the raw input data plus the complex data, the complex data produced by the fit,
        and all fit parameters with fit_axis as coordinate.
    """

    _z = array.amplitude.values * np.exp(1j * array.phase.values)
    z = DataArray(_z, name='complex', coords={fit_axis: getattr(array, fit_axis), 'frequency': array.frequency},
                  dims=[fit_axis, 'frequency'])

    array = merge([array, z])

    fitresults = np.zeros((getattr(array, fit_axis).shape[0], 15))
    fit_data = np.empty_like(array.complex.values)

    for i in range(getattr(array, fit_axis).shape[0]):

        z_dat = array.complex.isel({fit_axis: [i]}).values[0]

        res_fit = notch_port(f_data=array.frequency.values, z_data_raw=z_dat)
        res_fit.autofit()
        fitresults[i, :] = list(res_fit.fitresults.values())

        fit_data[i, :] = res_fit.z_data_sim

        if plot_fit:
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
            ax2.plot(array.frequency.values, 20 * np.log10(np.abs(z_dat)))
            ax2.plot(array.frequency.values, 20 * np.log10(np.abs(res_fit.z_data_sim)), color='blue')

            ax2_2 = ax2.twinx()
            ax2_2.plot(array.frequency.values, np.angle(z_dat), color='tab:orange')
            ax2_2.plot(array.frequency.values, np.angle(res_fit.z_data_sim), color='orange')

    _fxA = []
    for i in range(fitresults.shape[1]):
        _fxA += [DataArray(fitresults[:, i], name=list(res_fit.fitresults.keys())[i],
                           coords={fit_axis: getattr(array, fit_axis).values}, dims=[fit_axis])]

    _zfitxA = DataArray(fit_data, name='complex_fit', coords={fit_axis: getattr(array, fit_axis),
                                                              'frequency': array.frequency},
                        dims=[fit_axis, 'frequency'])

    array = merge([array, *_fxA, _zfitxA])

    return array
