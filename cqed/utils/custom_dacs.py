import qcodes as qc
from qcodes.instrument import parameter, base
import numpy as np
from functools import partial

class TransformationInstrument(base.Instrument):
    def __init__(self, name, parameter_names, parameter_units, subject_parameters, multiplier_matrix, fixed_indices, fixed_values):
        super().__init__(name)
        self.parameter_names = parameter_names
        self.parameter_units = parameter_units
        self.subject_parameters = subject_parameters
        self.set_matrix(multiplier_matrix)
        self.fixed_indices = fixed_indices
        self.fixed_values = fixed_values

        for index, (pname, unit) in enumerate(zip(parameter_names, parameter_units)):
            self.add_parameter(name=f'{pname}',
                               label=f'{pname} value',
                               unit=unit,
                               get_cmd=partial(self._get_component, index),
                               set_cmd=partial(self._set_component, index))

    def set_matrix(self, multiplier_matrix):
        if np.abs(np.linalg.det(multiplier_matrix)) < 1e-8:
            raise Exception("determinant of matrix too close to zero")
        self.multiplier_matrix = multiplier_matrix
        self.inverse_multiplier_matrix = np.linalg.inv(multiplier_matrix)
              
    def transformed_to_subject(self, values):
        return np.dot(self.inverse_multiplier_matrix, values).tolist()
    
    def subject_to_transformed(self, values):
        return np.dot(self.multiplier_matrix, values).tolist()
    
    def _get_components(self):
        components = []
        for p in self.subject_parameters:
            components.append(p())
        values = np.dot(self.multiplier_matrix, components).tolist()
        return values

    def _get_component(self, index):
        values = self._get_components()
        return values[index]

    def _set_component(self, index, value):
        current_values = self._get_components()
        current_values[index] = value
        if self.fixed_indices is not None:
            for ii in self.fixed_indices:
                current_values[ii] = self.fixed_values[ii]
        set_values = np.dot(self.inverse_multiplier_matrix, current_values)
        for p, v in zip(self.subject_parameters, set_values):
            p(v)