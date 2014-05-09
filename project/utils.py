__author__ = 'Federico Vaggi'

import numpy as np


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




