from xarray import DataArray, merge
from resonator_tools.circuit import notch_port
import numpy as np
import matplotlib.pyplot as plt
from resonator import shunt, background, see


def fit_resonator(array, fit_axis, fit_code='both', plot_fit=False, background_model='MagnitudePhase'):
    """
    Takes an xarray with data variables called 'amplitude' and 'phase' and returns an xarray consisting of the original
    raw data and the resonator fit parameters as a function of the coordinate 'fit_axis' (which is one of the
    coordinates of the input xarray). Fitting can be performed using the 'notch_port' class of the circle-fit routine
    provided by https://github.com/sebastianprobst/resonator_tools or the linear shunt model of
    https://github.com/danielflanigan/resonator.
    @param array: xarray with the data variables amplitude (linear units) and phase (radians), has to have the
        coordinate frequency, and at least one further coordinate
    @param fit_axis: coordinate of the xarray along which the fits should be performed
    @param fit_code: 'both', 'probst', 'flanigan' pick which fitting code to use
    @param plot_fit: If True shows all raw data with fits overlay
    @param flanigan_params: dictionary of parameters to set up the fitting model in the Flanigan code
    @return: xarray consisting of the raw input data plus the complex data, the complex data produced by the fit,
        and all fit parameters with fit_axis as coordinate.
    """
    # set the array dimensions for flanigan fit
    _dims_dict = {'Magnitude': 5, 'MagnitudePhase': 6, 'MagnitudePhaseDelay': 8, 'MagnitudeSlopeOffsetPhaseDelay': 9}

    if (fit_code != 'probst') and (background_model not in _dims_dict.keys()):
        print(f'background model must be one of {_dims_dict.keys():} fitting only with probst algorithm')
        fit_code = 'probst'

    _z = array.amplitude.values * np.exp(1j * array.phase.values)
    z = DataArray(_z, name='complex', coords={fit_axis: getattr(array, fit_axis), 'frequency': array.frequency},
                  dims=[fit_axis, 'frequency'])

    array = merge([array, z])

    # ToDo: make looping through fit_axis outermost loop

    if (fit_code == 'probst') or (fit_code == 'both'):

        fitresults_probst = np.zeros((getattr(array, fit_axis).shape[0], 15))
        fit_data = np.empty_like(array.complex.values)

        for i in range(getattr(array, fit_axis).shape[0]):

            z_dat = array.complex.isel({fit_axis: [i]}).values[0]

            res_fit = notch_port(f_data=array.frequency.values, z_data_raw=z_dat)
            res_fit.autofit()
            fitresults_probst[i, :] = list(res_fit.fitresults.values())

            fit_data[i, :] = res_fit.z_data_sim

            if plot_fit:
                fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
                ax2.plot(array.frequency.values, 20 * np.log10(np.abs(z_dat)))
                ax2.plot(array.frequency.values, 20 * np.log10(np.abs(res_fit.z_data_sim)), color='blue')

                ax2_2 = ax2.twinx()
                ax2_2.plot(array.frequency.values, np.angle(z_dat), color='tab:orange')
                ax2_2.plot(array.frequency.values, np.angle(res_fit.z_data_sim), color='orange')

        _fxA = []
        for i in range(fitresults_probst.shape[1]):
            _fxA += [DataArray(fitresults_probst[:, i], name='probst_'+list(res_fit.fitresults.keys())[i],
                               coords={fit_axis: getattr(array, fit_axis).values}, dims=[fit_axis])]

        _zfitxA = DataArray(fit_data, name='complex_fit', coords={fit_axis: getattr(array, fit_axis),
                                                                  'frequency': array.frequency},
                            dims=[fit_axis, 'frequency'])

        array = merge([array, *_fxA, _zfitxA])

    if (fit_code == 'flanigan') or (fit_code == 'both'):

        fitresults_flanigan = np.zeros((getattr(array, fit_axis).shape[0], _dims_dict[background_model] + 2))
        fitdicts = np.zeros((getattr(array, fit_axis).shape[0]), dtype=object)
        for i in range(getattr(array, fit_axis).shape[0]):

            z_dat = array.complex.isel({fit_axis: [i]}).values[0]
            model_flanigan = shunt.LinearShuntFitter(frequency=array.frequency.values, data=z_dat,
                                                     background_model=getattr(background, background_model)())
            fitresults_flanigan[i, :-2] = list(model_flanigan.result.values.values())
            fitresults_flanigan[i, -2] = model_flanigan.result.params['internal_loss'].stderr
            fitresults_flanigan[i, -1] = model_flanigan.result.params['coupling_loss'].stderr
            fitdicts[i] = model_flanigan.result

            if plot_fit:
                pass
                # ToDo add plot function

        _fxB = []
        flan_names = list(model_flanigan.result.values.keys()) + ['internal_loss_err'] + ['coupling_loss_err']
        # ToDo: transform coupling loss and internal loss and corresponding errors into Qi before saving?
        for ind, var_name in enumerate(flan_names):
            _fxB += [DataArray(fitresults_flanigan[:, ind], name='flan_' + var_name,
                               coords={fit_axis: getattr(array, fit_axis).values}, dims=[fit_axis])]

        array = merge([array, *_fxB])


    return array
