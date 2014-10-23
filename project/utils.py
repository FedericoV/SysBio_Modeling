__author__ = 'Federico Vaggi'

import numpy as np
from pandas import MultiIndex, concat, DataFrame

###############################################################################
# Simple mapping functions
###############################################################################


def direct_model_var_to_measure(model_sim, measurement, mapping_parameters):
    """"
    Given a vector of simulated species model_sim, and the timepoints at which it's simulated model_timepoints,
    and a vector of measurements and the measurement of the timepoints, it maps the species to the measurement"""
    var_model_idx = mapping_parameters  # What's the simulation variable that maps to our measurement
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_sim['timepoints'], measure_timepoints)
    mapped_df = model_sim.iloc[exp_t_idx, (var_model_idx, -1)]  # -1 is to also capture the timepoints

    new_columns = mapped_df.columns.values
    new_columns[0] = 'values'
    mapped_df.columns = new_columns
    return mapped_df


def direct_model_jac_to_measure_jac(model_jacobian, measurement, mapping_parameters):
    model_idx = mapping_parameters

    col_str = ",".join(model_jacobian.columns.tolist())
    n_exp_params = col_str.count("dy0/")
    # Count with respect to how many parameters is a variable differentiated

    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_jacobian['timepoints'], measure_timepoints)

    # Mapping between experimental measurement and model variable
    v = model_idx * n_exp_params

    mapped_jac = model_jacobian.iloc[exp_t_idx, v:(v + n_exp_params)].copy()
    mapped_jac['timepoints'] = model_jacobian['timepoints'].values[exp_t_idx].copy()
    return mapped_jac


def sum_model_vars_to_measure(model_sim, measurement, mapping_parameters):
    model_variable_idxs = mapping_parameters
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_sim['timepoints'], measure_timepoints)

    mapped_df = np.zeros((len(exp_t_idx),))
    for v in model_variable_idxs:
        mapped_df += model_sim.iloc[exp_t_idx, v]

    mapped_df = DataFrame(mapped_df, columns=['values'])
    mapped_df['timepoints'] = model_sim['timepoints'].values[exp_t_idx].copy()

    return mapped_df


def sum_model_jac_to_measure_jac(model_jacobian, measurement, mapping_parameters):
    model_variable_idxs = mapping_parameters
    jac_columns = model_jacobian.columns.tolist()

    col_str = ",".join(jac_columns)
    n_exp_params = col_str.count("dy0/")
    # Count with respect to how many parameters is a variable differentiated

    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_jacobian['timepoints'], measure_timepoints)

    mapped_jac = np.zeros((len(exp_t_idx), n_exp_params))
    for v in model_variable_idxs:
        v *= n_exp_params
        mapped_jac += model_jacobian[exp_t_idx, v:(v + n_exp_params)]

    mapped_jac = DataFrame(mapped_jac, columns=jac_columns[:-1])
    mapped_jac['timepoints'] = model_jacobian['timepoints'].values[exp_t_idx].copy()

    return mapped_jac


###############################################################################
# Ordered Hash Dict
###############################################################################
from collections import OrderedDict


class OrderedHashDict(OrderedDict):
    def __getitem__(self, key):
        try:
            _v = super(OrderedHashDict, self).__getitem__(key)
            return _v
        except KeyError:
            for _k, _v in self.items():
                if type(_k) == str:
                    continue
                elif key in _k:
                    return _v
            raise KeyError("%s not in dictionary or in any of the groups in the dictionary")

    def __contains__(self, key):
        try:
            self[key]
            return True
        except KeyError:
            return False

    def __setitem__(self, key, value):
        if (type(key) != str) and (type(key) != frozenset):
            raise TypeError("Keys can only be strings, or frozen sets of strings")

        if type(key) == frozenset:
            if key in self:
                super(OrderedHashDict, self).__setitem__(key, value)

            else:
                for k in key:
                    if k in self:
                        hashgroup = self[k]
                        raise KeyError("%s already in dict in a hashgroup" % key)
                super(OrderedHashDict, self).__setitem__(key, value)

        else:  # It's a string
            if key not in self:
                super(OrderedHashDict, self).__setitem__(key, value)

            else:  # The string is the dict, either as a hashgroup or as a key
                found = False
                for _k in self.keys():
                    if _k == key:
                        super(OrderedHashDict, self).__setitem__(key, value)
                        found = True

                if not found:
                    raise KeyError("%s already in dict in a hashgroup" % key)


def exp_param_transform(project_param_vector):
    """
    Sometimes, it's convenient to optimize models in log-space to avoid negative values.
    Instead of doing :math:`Y_{sim}(\\theta)` we compute :math:`Y_{sim}(f(\\theta))`

    Parameters
    ----------
    project_param_vector: :class:`~numpy:numpy.ndarray`
        An (n,) dimensional array containing the parameters being optimized in the project

    Returns
    -------
    transformated_parameters: :class:`~numpy:numpy.ndarray`
        An (n,) dimensional array the parameters after applying a transformation

    See Also
    --------
    param_transform_derivative

    """
    exp_param_vector = np.exp(project_param_vector)
    return exp_param_vector


def exp_param_transform_derivative(project_param_vector):
    """
    The derivative of the function applied to the parameters prior to the simulation.
    :math:`\\frac{\\partial f}{\\partial \\theta}`

    Parameters
    ----------
    project_param_vector: :class:`~numpy:numpy.ndarray`
        An (n,) dimensional array containing the parameters being optimized in the project

    Returns
    -------
    transformation_derivative: :class:`~numpy:numpy.ndarray`
        An (n,) dimensional array  containing the derivatives of the parameter transformation function

    See Also
    --------
    param_transform
    """
    transformation_derivative = np.exp(project_param_vector)
    return transformation_derivative