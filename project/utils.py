__author__ = 'Federico Vaggi'

import numpy as np

###############################################################################
# Simple mapping functions
###############################################################################


def direct_model_var_to_measure(model_sim, model_timepoints, experiment, measurement, mapping_parameters):
    model_idx = mapping_parameters
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_timepoints, measure_timepoints)
    return model_sim[exp_t_idx, model_idx], exp_t_idx


def direct_model_jac_to_measure_jac(model_jacobian, model_timepoints, experiment, measurement, mapping_parameters):
    model_idx = mapping_parameters
    n_exp_params = len(experiment.param_global_vector_idx)
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_timepoints, measure_timepoints)

    # Mapping between experimental measurement and model variable
    v = model_idx * n_exp_params

    return model_jacobian[exp_t_idx, v:(v + n_exp_params)]


def sum_model_vars_to_measure(model_sim, model_timepoints, experiment, measurement, mapping_parameters):
    model_variable_idxs = mapping_parameters
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_timepoints, measure_timepoints)

    measure_sim = np.zeros((len(exp_t_idx),))
    for v in model_variable_idxs:
        measure_sim += model_sim[exp_t_idx, v]

    return measure_sim, exp_t_idx


def sum_model_jac_to_measure_jac(model_jacobian, model_timepoints, experiment, measurement, mapping_parameters):
    model_variable_idxs = mapping_parameters
    n_exp_params = len(experiment.param_global_vector_idx)
    _, _, measure_timepoints = measurement.get_nonzero_measurements()
    exp_t_idx = np.searchsorted(model_timepoints, measure_timepoints)

    measure_jac = np.zeros((len(exp_t_idx), n_exp_params))
    for v in model_variable_idxs:
        v *= n_exp_params
        measure_jac += model_jacobian[exp_t_idx, v:(v + n_exp_params)]

    return measure_jac


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
                for _k in self.keys():
                    if _k == key:
                        super(OrderedHashDict, self).__setitem__(key, value)

                raise KeyError("%s already in dict in a hashgroup" % key)


