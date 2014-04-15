from collections import OrderedDict, defaultdict
import warnings

import numpy as np


########################################################################################
# Utility Functions
########################################################################################


def _accumulate__scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim):
    sim_dot_exp[:] += np.sum(((exp_data/exp_std**2) * sim_data))
    sim_dot_sim[:] += np.sum(((sim_data/exp_std) * (sim_data/exp_std)))


def _accumulate__scale_factors_jac(exp_data, exp_std, sim_data, model_sens,
                                   sim_dot_exp, sim_dot_sim, sens_dot_exp_data, sens_dot_sim):
    sens_dot_exp_data[:] += np.sum(model_sens.T*exp_data / (exp_std**2), axis=1)  # Vector
    sens_dot_sim[:] += np.sum(model_sens.T*sim_data / (exp_std**2), axis=1)  # Vector
    sim_dot_sim[:] += np.sum((sim_data * sim_data) / (exp_std**2))  # Scalar
    sim_dot_exp[:] += np.sum((sim_data * exp_data) / (exp_std**2))  # Scalar


def _combine__scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp, scale_jac_out):
    scale_jac_out[:] = (sens_dot_exp_data/sim_dot_sim - 2*sim_dot_exp*sens_dot_sim/sim_dot_sim**2)


class SimpleProject(object):
    """Class to simulate experiments with a given model

    Attributes
    ----------
    model : :class:`~OdeModel:SysBio_Modeling.model.ode_model.OdeModel`
        A model that can simulate experiments
    experiments: list of experiments
        A sorted collection of experiments.
    n_vars : int
        The number of state variables in the model
    param_order : list
        Order in which the parameters appear in the model.  Important for returning the jacobian in the correct
        order.
    use_jit : bool , optional
        Whether or not to jit `sens_model` and `model` using numba
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
        self.global_param_idx, self.n_global_params, self.residuals_per_param = self._set_local_param_idx()

        self._n_residuals = self._calc__n_residuals()
        self._measurements_idx = self._set_measurement_idx()
        self._all_sims = None
        self._all_residuals = None
        self._model_jacobian = None
        self._scale_factors = {}
        self._scale_factors_jacobian = {}
        self.global_param_vector = None

    def reset_calcs(self):
        self._all_sims = None
        self._all_residuals = None
        self._model_jacobian = None
        self._scale_factors = {}
        self._scale_factors_jacobian = {}
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

    def _calc__n_residuals(self, include_zero_timepoints=False):
        """
        Calculates the total number of experimental points across all experiments
        """
        n_res = 0
        for experiment in self.experiments:
            for measured_var, measurement in experiment.measurements.items():
                timepoints = measurement['timepoints']
                if not include_zero_timepoints:
                    timepoints = timepoints[timepoints != 0]
                n_res += len(timepoints)
        return n_res

    def _set_local_param_idx(self):
        """
        We map the model parameters to the global parameters using the model parameter settings dictionary.

        Parameters come in 3 flavors:

        - Local: A parameter is local if it is allowed to take on a different value in each experiment
        - Global: A parameter is global if it has a fixed value across all experiments
        - Shared: Shared parameters belong to groups, and each group depends on a number of settings.  All parameters
        in the same group will have the same value in experiments that have the same settings.

        Typically, the number of global parameters will be much bigger than the number of model parameters.
        """

        global_pars = self.model_parameter_settings.get('Global', [])
        fully_local_pars = self.model_parameter_settings.get('Local', {})
        shared_pars = self.model_parameter_settings.get('Shared', {})
        fixed_pars = self.model_parameter_settings.get('Fixed', [])

        global_param_idx = {}
        # This dict maps the location of parameters in the global parameter vector.
        residuals_per_param = defaultdict(lambda: defaultdict(lambda: 0))

        n_params = 0
        for p in global_pars:
            global_param_idx[p] = {}
            global_param_idx[p]['Global'] = n_params
            n_params += 1

        for exp_idx, experiment in enumerate(self.experiments):
            exp_param_idx = OrderedDict()

            for p in global_pars:
                if p not in fixed_pars:
                    exp_param_idx[p] = global_param_idx[p]['Global']
                    residuals_per_param[p]['Global'] += len(experiment.get_unique_timepoints())
                # Global parameters always refer to the same index in the global parameter vector

            for p_group in shared_pars:
                # Shared parameters depend on the experimental settings.
                for p, settings in shared_pars[p_group].items():
                    settings = tuple(settings)

                    exp_p_settings = tuple([experiment.settings[setting] for setting in settings])
                    # Here we get the experimental conditions for all settings upon which that parameter depends.

                    if p_group not in global_param_idx:
                        global_param_idx[p_group] = {}
                    # If it's the first time we encounter that parameter group, create it.

                    try:
                        exp_param_idx[p] = global_param_idx[p_group][exp_p_settings]
                        # If we already have another parameter in that parameter group and with those same
                        # experimental settings, then they point to the same location.
                    except KeyError:
                        global_param_idx[p_group][exp_p_settings] = n_params
                        exp_param_idx[p] = n_params
                        n_params += 1
                    residuals_per_param[p_group][exp_p_settings] += len(experiment.get_unique_timepoints())

            for p in fully_local_pars:
                par_string = '%s_%s' % (p, experiment.name)
                global_param_idx[par_string]['Local'] = n_params
                exp_param_idx[p] = n_params
                n_params += 1
                residuals_per_param[par_string]['Local'] += len(experiment.get_unique_timepoints())

            for p in fixed_pars:
                assert(p in experiment.fixed_parameters)
                # If some parameters are marked as globally fixed, each experiment must provide a value.

            self.experiments[exp_idx].param_global_vector_idx = exp_param_idx
        return global_param_idx, n_params, residuals_per_param

    def _sim_experiments(self, exp_subset='all', all_timepoints=False):
        """
        Simulates all the experiments in the project.
        """

        if self.global_param_vector is None:
            raise ValueError('Parameter vector not set')
        _all_sims = []
        if exp_subset is 'all':
            exp_subset = self.experiments
        else:
            exp_subset = self.experiments[exp_subset]

        for experiment in exp_subset:
            exp_sim = self.model.simulate_experiment(self.global_param_vector, experiment,
                                                     self.measurement_variable_map, all_timepoints)
            _all_sims.append(exp_sim)
        self._all_sims = _all_sims

    def _calc__scale_factors(self):
        """
        Analytically calculates the optimal scale factor for measurements that are in arbitrary units
        """
        for measure_name, experiment_list in self._measurements_idx.items():
            sim_dot_exp = np.zeros((1,), dtype='float64')
            sim_dot_sim = np.zeros((1,), dtype='float64')

            for exp_idx in experiment_list:
                experiment = self.experiments[exp_idx]
                measurement = experiment.measurements[measure_name]
                exp_timepoints = measurement['timepoints']

                exp_data = measurement['value'][exp_timepoints != 0]
                exp_std = measurement['std_dev'][exp_timepoints != 0]

                sim_data = self._all_sims[exp_idx][measure_name]['value']
                _accumulate__scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim)

            self._scale_factors[measure_name] = sim_dot_exp / sim_dot_sim

    def _calc__scale_factors_jacobian(self):
        """
        Analytically calculates the jacobian of the scale factors for each measurement
        """
        n_global_pars = len(self.global_param_vector)
        for measure_name, experiment_list in self._measurements_idx.items():
            sens_dot_exp_data = np.zeros((n_global_pars,), dtype='float64')
            sens_dot_sim = np.zeros((n_global_pars,), dtype='float64')
            sim_dot_sim = np.zeros((1,), dtype='float64')
            sim_dot_exp = np.zeros((1,), dtype='float64')

            for exp_idx in experiment_list:
                experiment = self.experiments[exp_idx]
                measurement = experiment.measurements[measure_name]
                exp_data = measurement['value']
                exp_std = measurement['std_dev']

                exp_timepoints = measurement['timepoints']
                exp_data = exp_data[exp_timepoints != 0]  # Vector
                exp_std = exp_std[exp_timepoints != 0]  # Vector

                sim_data = self._all_sims[exp_idx][measure_name]['value']  # Vector
                model_sens = self._model_jacobian[exp_idx][measure_name]  # Matrix

                _accumulate__scale_factors_jac(exp_data, exp_std, sim_data, model_sens, sim_dot_exp, sim_dot_sim,
                                               sens_dot_exp_data, sens_dot_sim)

            scale_jac_out = np.zeros((n_global_pars,), dtype='float64')
            _combine__scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp, scale_jac_out)
            self._scale_factors_jacobian[measure_name] = scale_jac_out

    def _calc_exp_residuals(self, exp_idx):
        """Returns the residuals between simulated data and the
        experimental data.

        Residuals are |sim(model, param)_i - exp_data_i|
        """
        residuals = OrderedDict()
        experiment = self.experiments[exp_idx]
        for measure_name in experiment.measurements:
            try:
                scale = self._scale_factors[measure_name]
            except KeyError:
                scale = 1
            measurement = experiment.measurements[measure_name]
            exp_timepoints = measurement['timepoints']
            exp_data = measurement['value'][exp_timepoints != 0]
            exp_std = measurement['std_dev'][exp_timepoints != 0]
            sim_data = self._all_sims[exp_idx][measure_name]['value']

            residuals[measure_name] = (sim_data * scale - exp_data) / exp_std
        return residuals

    def _calc_residuals(self):
        """
        Calculates residuals between the simulations, scaled by the scaling factor and the simulations.
        Has to be called after simulations and scaling factors are calculated
        """
        _all_residuals = []
        for exp_idx, experiment in enumerate(self.experiments):
            exp_res = self._calc_exp_residuals(exp_idx)
            _all_residuals.append(exp_res)
        self._all_residuals = _all_residuals

    def _calc__model_jacobian(self):
        all_jacobians = []
        for experiment in self.experiments:
            if self.global_param_vector is None:
                raise ValueError('Parameter vector not set')
            exp_jacobian = self.model.calc_jacobian(self.global_param_vector,
                                                    experiment, self.measurement_variable_map)
            all_jacobians.append(exp_jacobian)
        self._model_jacobian = all_jacobians

    def print_param_settings(self, verbose=True):
        """
        Prints out all the parameters combinations in the project in a fancy way
        """

        total_params = 0
        for p_group in self.global_param_idx:
            exp_settings = self.global_param_idx[p_group].keys()
            exp_settings = sorted(exp_settings)
            if verbose:
                print '%s  total_settings: %d ' % (p_group, len(exp_settings))
            for exp_set in exp_settings:
                if verbose:
                    print '%s, %d \t' % (repr(exp_set), self.residuals_per_param[p_group][exp_set])
                total_params += 1
            if verbose:
                print '\n***********************'
        if verbose:
            print "Total Parameters: %d" % total_params
        return total_params

    def __call__(self, global_param_vector):
        """
        Calculates the residuals between the simulated values (after optimal scaling) and the experimental values
        across all experiments in the project.

        :rtype: :class:`~numpy:numpy.ndarray`
        :param np.array global_param_vector: A vector of all parameter values that aren't fixed
        :return: A vector of residuals calculated for all measures in all experiments.
        """
        self.reset_calcs()
        self.global_param_vector = np.copy(global_param_vector)
        self._sim_experiments()
        self._calc__scale_factors()
        self._calc_residuals()

        residual_array = np.zeros((self._n_residuals,))
        res_idx = 0
        for exp_res in self._all_residuals:
            for res_block in exp_res.values():
                residual_array[res_idx:res_idx+len(res_block)] = res_block
                res_idx += len(res_block)
        return residual_array

    def global_jacobian(self, global_param_vector):
        """
        We are minimizing:

        .. math::
            C(\\theta)= 0.5*(\\sum{B*Y(\\theta)_{sim} - Y_{exp}})^2

        Where:
         :math:`B` are the optimal scaling factors for each measure \n
         :math:`\\theta` are the parameters that we are optimizing \n
         :math:`Y(\\theta)_{sim}` is the output of the model as a function of the parameters \n
         :math:`Y_{exp}` is the experimental data \n

        \\frac{\\partial {BY_{sim}}}{\\partial \\theta}


        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            Vector containing all parameters being optimized across the project

        Returns
        -------
        global_jacobian: :class:`~numpy:numpy.ndarray`
            An (n, m) array where n is the number of residuals and m is the number of global parameters.
        """

        self.reset_calcs()
        self.global_param_vector = np.copy(global_param_vector)
        self._sim_experiments()
        self._calc__scale_factors()
        self._calc_residuals()

        self._calc__model_jacobian()
        self._calc__scale_factors_jacobian()

        jacobian_array = np.zeros((self._n_residuals, len(global_param_vector)))
        res_idx = 0
        for exp_jac, exp_sim in zip(self._model_jacobian, self._all_sims):
            for measure in exp_jac:
                measure_sim = exp_sim[measure]['value']
                measure_sim_jac = exp_jac[measure]
                measure_scale = self._scale_factors[measure]
                measure_scale_jac = self._scale_factors_jacobian[measure]
                jac = measure_sim_jac*measure_scale + measure_scale_jac.T*measure_sim[:, np.newaxis]
                jacobian_array[res_idx:res_idx+len(measure_sim), :] = jac
                res_idx += len(measure_sim)

        return jacobian_array

    def flat_jacobian(self, global_param_vector):
        """
        Returns the gradient of the cost function.

        The cost function is:
        .. math::
            C(\\theta)= 0.5*(\\sum{B*Y(\\theta)_{sim} - Y_{exp}})^2

        This function returns :math:`\\frac{\\partial C}}{\\partial \\theta}`

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            Vector containing all parameters being optimized across the project

        Returns
        -------
        global_jacobian: :class:`~numpy:numpy.ndarray`
            An (m,) dimensional array.
        """
        jacobian_matrix = self.global_jacobian(global_param_vector)
        residuals = self(global_param_vector)
        return (jacobian_matrix.T * residuals).sum(axis=1)

    def sum_square_residuals(self, global_param_vector):
        """
        Returns the sum squared residuals across all experiments

        Returns:
        .. math::
            C(\\theta)= 0.5*(\\sum{B*Y(\\theta)_{sim} - Y_{exp}})^2

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            Vector containing all parameters being optimized across the project

        Returns
        -------
        rss: : float
            The sum of all residuals
        """
        residuals = self(global_param_vector)
        return 0.5*np.sum(residuals**2)

    def nlopt_fcn(self, global_param_vector, grad):
        """
        Wrapper to make objective function compatible with nlopt API

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            Vector containing all parameters being optimized across the project

        grad: :class:`~numpy:numpy.ndarray`, optional
            Cost function gradient.  Only used for gradient based optimization by nlopt

        Returns
        -------
        rss : float
            The sum of all residuals
        """
        if grad.size > 0:
            grad[:] = self.flat_jacobian(global_param_vector)
        return self.sum_square_residuals(global_param_vector)

    def load_param_dict(self, param_dict, default_value=0.0):
        """
        Loads parameters from a dictionary (default output format for saving at the end of a fit)

        Parameters
        ----------
        param_dict: dict
            Dictionary of parameters

        default_value: float
            Value for the global_parameter_vector in case it's not present in the param_dict

        Returns
        -------
        param_vector: :class:`~numpy:numpy.ndarray`
            An array with the parameter values of param_dict in the correct index.

        See Also
        --------
        param_vect_to_dict
        """
        param_vector = np.ones((self.n_global_params,)) * default_value
        for p_group in self.global_param_idx:
            for exp_settings, global_idx in self.global_param_idx[p_group].items():
                try:
                    param_vector[global_idx] = param_dict[p_group][exp_settings]
                except KeyError:
                    pass
        return param_vector

    def param_vect_to_dict(self, param_vector):
        """
        Loads parameters from a vector into a dictionary structure

        Parameters
        ----------
        param_vector: :class:`~numpy:numpy.ndarray`
            An array of parameters

        Returns
        -------
        param_vector: :class:`~numpy:numpy.ndarray`
            Dictionary of parameters

        See Also
        --------
        load_param_dict
        """
        param_dict = {}
        for p_group in self.global_param_idx:
            param_dict[p_group] = {}
            for exp_settings, global_idx in self.global_param_idx[p_group].items():
                param_dict[p_group][exp_settings] = param_vector[global_idx]
        return param_dict
