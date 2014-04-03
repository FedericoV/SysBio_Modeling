from collections import OrderedDict
import warnings

import numpy as np


class SimpleProject(object):
    """
    Project with one kind of model and identical mapping for experiment
    to model
    """

    def __init__(self, model, experiments, model_parameter_settings,
                 measurement_variable_map):
        self.model = model
        if type(experiments) is not list:
            warnings.warn('Make sure that the iterable passed has a stable iteration order')
        self.experiments = experiments
        # We expect that we always iterate through experiments in same order.

        self.model_parameter_settings = model_parameter_settings
        self.measurement_variable_map = measurement_variable_map
        self.global_param_idx, self.n_global_params = self._set_local_param_idx()

        self.n_residuals = self._calc_n_residuals()
        self.measurements_idx = self._set_measurement_idx()

        self.all_sims = None
        self.all_residuals = None
        self.model_jacobian = None
        self.scale_factors = {}
        self.scale_factors_jacobian = {}
        self.global_param_vector = None

    def reset_calcs(self):
        self.all_sims = None
        self.all_residuals = None
        self.model_jacobian = None
        self.scale_factors = {}
        self.scale_factors_jacobian = {}
        self.global_param_vector = None

    def _set_measurement_idx(self):
        m_idx = {}
        for i, experiment in enumerate(self.experiments):
            for measured_var in experiment.measurements:
                try:
                    m_idx[measured_var].append(i)
                except KeyError:
                    m_idx[measured_var] = []
                    m_idx[measured_var].append(i)
        return m_idx

    def _calc_n_residuals(self, include_zero_timepoints=False):
        """Just a counter that keeps track of the total number of residuals"""
        n_res = 0
        for experiment in self.experiments:
            for measured_var, measurement in experiment.measurements.items():
                timepoints = measurement['timepoints']
                if not include_zero_timepoints:
                    timepoints = timepoints[timepoints != 0]
                n_res += len(timepoints)
        return n_res

    def _set_local_param_idx(self):
        # This dict holds the index of where in the parameter vector
        # we find the the value of a parameter for a given setting.
        # Global parameters will be independent of experimental settings.
        # Local parameters will be dependent on the settings of each experiment.
        # Experiments with the same setting should have the same index value for
        # the parameter.

        global_pars = self.model_parameter_settings.get('Global', {})
        fully_local_pars = self.model_parameter_settings.get('Local', {})
        shared_pars = self.model_parameter_settings.get('Shared', {})

        global_param_idx = {}
        n_params = 0
        for p in global_pars:
            global_param_idx[p] = {}
            global_param_idx[p]['Global'] = n_params
            n_params += 1

        for exp_idx, exp in enumerate(self.experiments):
            # For each experiment, where in the full parameter vector we find
            # the parameter we want to optimize.  Note - the position should be
            # the same as that found in the exp_param_idx.  The full parameter
            # vector, however, will use the global names for parameters.
            # Look at test at bottom to clarify.
            exp_param_idx = OrderedDict()

            for p in global_pars:
                exp_param_idx[p] = global_param_idx[p]['Global']

            for p_group in shared_pars:
                for p, settings in shared_pars[p_group].items():
                    settings = tuple(settings)

                    # Local parameters depend on specific experimental settings.
                    # Note that ('L', 'R') != ('R', 'L')
                    exp_p_settings = tuple([exp.settings[setting] for
                                            setting in settings])

                    if p_group not in global_param_idx:
                        global_param_idx[p_group] = {}

                    # If that combination of settings is already present,
                    # we point it to the already existing index.
                    try:
                        exp_param_idx[p] = global_param_idx[p_group][exp_p_settings]
                    except KeyError:
                        global_param_idx[p_group][exp_p_settings] = n_params
                        exp_param_idx[p] = n_params
                        n_params += 1

            for p in fully_local_pars:
                global_param_idx['%s_%s' % (p, exp.name)]['Local'] = n_params
                exp_param_idx[p] = n_params
                n_params += 1

            self.experiments[exp_idx].param_global_vector_idx = exp_param_idx
        return global_param_idx, n_params

    def _sim_experiments(self, exp_subset='all', all_timepoints=False):
        """

        :rtype list[dict()]
        """

        if self.global_param_vector is None:
            raise ValueError('Parameter vector not set')
        all_sims = []
        if exp_subset is 'all':
            exp_subset = self.experiments
        else:
            exp_subset = self.experiments[exp_subset]

        for experiment in exp_subset:
            exp_sim = self.model.simulate_experiment(self.global_param_vector, experiment,
                                                     self.measurement_variable_map, all_timepoints)
            all_sims.append(exp_sim)
        self.all_sims = all_sims

    def _calc_scale_factors(self):
        """
        Call only after _sim_experiments
        """
        for measure_name, experiment_list in self.measurements_idx.items():
            sim_dot_exp = 0.0
            sim_dot_sim = 0.0

            for exp_idx in experiment_list:
                experiment = self.experiments[exp_idx]
                measurement = experiment.measurements[measure_name]
                exp_timepoints = measurement['timepoints']

                exp_data = measurement['value'][exp_timepoints != 0]
                exp_std = measurement['std_dev'][exp_timepoints != 0]

                sim_data = self.all_sims[exp_idx][measure_name]['value']
                sim_dot_exp += np.sum(((exp_data/exp_std**2) * sim_data))
                sim_dot_sim += np.sum(((sim_data/exp_std) * (sim_data/exp_std)))

            self.scale_factors[measure_name] = sim_dot_exp / sim_dot_sim

    def _calc_exp_residuals(self, exp_idx):
        """Returns the residuals between simulated data and the
        experimental data.

        Residuals are |sim(model, param)_i - exp_data_i|
        """
        residuals = OrderedDict()
        experiment = self.experiments[exp_idx]
        for measure_name in experiment.measurements:
            try:
                scale = self.scale_factors[measure_name]
            except KeyError:
                scale = 1
            measurement = experiment.measurements[measure_name]
            exp_timepoints = measurement['timepoints']
            exp_data = measurement['value'][exp_timepoints != 0]
            exp_std = measurement['std_dev'][exp_timepoints != 0]
            sim_data = self.all_sims[exp_idx][measure_name]['value']

            residuals[measure_name] = (sim_data * scale - exp_data) / exp_std
        return residuals

    def _calc_residuals(self):
        """Only call after _sim_experiments and
        :rtype : list[dict]
        _calc_scale_factors"""
        all_residuals = []
        for exp_idx, experiment in enumerate(self.experiments):
            exp_res = self._calc_exp_residuals(exp_idx)
            all_residuals.append(exp_res)
        self.all_residuals = all_residuals

    def _calc_model_jacobian(self):
        all_jacobians = []
        for experiment in self.experiments:
            if self.global_param_vector is None:
                raise ValueError('Parameter vector not set')
            exp_jacobian = self.model.calc_jacobian(self.global_param_vector,
                                                    experiment, self.measurement_variable_map)
            all_jacobians.append(exp_jacobian)
        self.model_jacobian = all_jacobians

    def _calc_scale_factors_jacobian(self):
        for measure_name, experiment_list in self.measurements_idx.items():
            sens_dot_exp_data = 0
            sens_dot_sim = 0
            sim_dot_sim = 0
            sim_dot_exp = 0

            for exp_idx in experiment_list:
                experiment = self.experiments[exp_idx]
                measurement = experiment.measurements[measure_name]
                exp_data = measurement['value']
                exp_std = measurement['std_dev']

                exp_timepoints = measurement['timepoints']
                exp_data = exp_data[exp_timepoints != 0]  # Vector
                exp_std = exp_std[exp_timepoints != 0]  # Vector

                sim_data = self.all_sims[exp_idx][measure_name]['value']  # Vector
                model_sens = self.model_jacobian[exp_idx][measure_name]  # Matrix

                sens_dot_exp_data += np.sum(model_sens.T*exp_data / (exp_std**2), axis=1)  # Vector

                sens_dot_sim += np.sum(model_sens.T*sim_data / (exp_std**2), axis=1)  # Vector
                sim_dot_sim += np.sum((sim_data * sim_data) / (exp_std**2))  # Scalar
                sim_dot_exp += np.sum((sim_data * exp_data) / (exp_std**2))  # Scalar

        self.scale_factors_jacobian[measure_name] = (sens_dot_exp_data/sim_dot_sim -
                                                     2*sim_dot_exp*sens_dot_sim/sim_dot_sim**2)

    def get_total_params(self, verbose=True):
        """
        Prints out the parameter index in a nice way
        """
        total_params = 0
        for g_param in self.global_param_idx:
            exp_settings = self.global_param_idx[g_param].keys()
            exp_settings = sorted(exp_settings)
            if verbose:
                print '%s  total_settings: %d ' % (g_param, len(exp_settings))
            for exp_set in exp_settings:
                if verbose:
                    print '\t' + repr(exp_set)
                total_params += 1
            if verbose:
                print '\n***********************'
        if verbose:
            print "Total Parameters: %d" % total_params
        return total_params

    def __call__(self, global_param_vector):
        """Here we use memoization"""
        self.reset_calcs()
        self.global_param_vector = np.copy(np.asarray(global_param_vector))
        self._sim_experiments()
        self._calc_scale_factors()
        self._calc_residuals()

        residual_array = np.zeros((self.n_residuals,))
        res_idx = 0
        for exp_res in self.all_residuals:
            for res_block in exp_res.values():
                residual_array[res_idx:res_idx+len(res_block)] = res_block
                res_idx += len(res_block)
        return residual_array

    def global_jacobian(self, global_param_vector):

        self.reset_calcs()
        self.global_param_vector = np.copy(np.asarray(global_param_vector))
        self._sim_experiments()
        self._calc_scale_factors()
        self._calc_residuals()

        self._calc_model_jacobian()
        self._calc_scale_factors_jacobian()

        jacobian_array = np.zeros((self.n_residuals, len(global_param_vector)))
        res_idx = 0
        for exp_jac, exp_sim in zip(self.model_jacobian, self.all_sims):
            for measure in exp_jac:
                measure_sim = exp_sim[measure]['value']
                measure_sim_jac = exp_jac[measure]
                measure_scale = self.scale_factors[measure]
                measure_scale_jac = self.scale_factors_jacobian[measure]
                jac = measure_sim_jac*measure_scale + measure_scale_jac.T*measure_sim[:, np.newaxis]
                jacobian_array[res_idx:res_idx+len(measure_sim), :] = jac
                res_idx += len(measure_sim)

        return jacobian_array

    def flat_jacobian(self, global_param_vector):
        # Optional factor of 2.
        jacobian_matrix = self.global_jacobian(global_param_vector)
        residuals = self(global_param_vector)
        return (jacobian_matrix.T * residuals).sum(axis=1)

    def sum_square_residuals(self, global_param_vector):
        residuals = self(global_param_vector)
        return 0.5*np.sum(residuals**2)

    def nlopt_fcn(self, global_param_vector, grad):
        if grad.size > 0:
            grad[:] = self.flat_jacobian()[:]
        return self.sum_square_residuals(global_param_vector)


    def load_param_dict(self, param_dict, default_value=0):
        param_vector = np.ones((self.n_global_params,)) * default_value
        for g_param in self.global_param_idx:
            for exp_set, idx in self.global_param_idx[g_param].items():
                try:
                    param_vector[idx] = param_dict[g_param][exp_set]
                except KeyError:
                    pass
        return param_vector

    def param_vect_to_dict(self, param_vector):
        param_dict = {}
        for g_param in self.global_param_idx:
            param_dict[g_param] = {}
            for exp_set, idx in self.global_param_idx[g_param].items():
                param_dict[g_param][exp_set] = param_vector[idx]
        return param_dict

