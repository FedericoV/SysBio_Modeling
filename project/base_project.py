from collections import OrderedDict, defaultdict
import warnings
import copy

import numpy as np
import scipy
import numba

import utils




########################################################################################
# Utility Functions
########################################################################################
@numba.jit
def _entropy_integrand(u, ak, bk, prior_B, sigma_log_B, T, B_best, log_B_best):
    """Copied from SloppyCell"""
    B_centered = np.exp(u) * B_best
    lB = u + log_B_best
    return np.exp(-ak / (2 * T) * (B_centered - B_best) ** 2 - (lB - prior_B) ** 2 / (2 * sigma_log_B ** 2))


def _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, exp_weight=1):
    sim_dot_exp[:] += np.sum(((exp_data/exp_std**2) * sim_data)) * exp_weight
    sim_dot_sim[:] += np.sum(((sim_data/exp_std) * (sim_data/exp_std))) * exp_weight


def _accumulate__scale_factors_jac(exp_data, exp_std, sim_data, model_sens,
                                   sim_dot_exp, sim_dot_sim, sens_dot_exp_data, sens_dot_sim, exp_weight=1):
    sens_dot_exp_data[:] += np.sum(model_sens.T*exp_data / (exp_std**2), axis=1) * exp_weight  # Vector
    sens_dot_sim[:] += np.sum(model_sens.T*sim_data / (exp_std**2), axis=1) * exp_weight  # Vector
    sim_dot_sim[:] += np.sum((sim_data * sim_data) / (exp_std**2)) * exp_weight  # Scalar
    sim_dot_exp[:] += np.sum((sim_data * exp_data) / (exp_std**2)) * exp_weight  # Scalar


def _combine__scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp, scale_jac_out):
    scale_jac_out[:] = (sens_dot_exp_data/sim_dot_sim - 2*sim_dot_exp*sens_dot_sim/sim_dot_sim**2)


class Project(object):
    """Class to simulate experiments with a given model

    Attributes
    ----------
    model : :class:`~OdeModel:SysBio_Modeling.model.ode_model.OdeModel`
        A model that can simulate experiments
    experiments: list
        A sorted collection of experiments.
    model_parameter_settings : dict
        How the parameters in the model vary depending on the experimental settings in the experiments
    measurement_to_model_map : dict
        A dictionary of functions that maps the variables simulated by the model to the observed
         measurements in the experiments.
    """

    def __init__(self, model, experiments, model_parameter_settings, measurement_to_model_map):
        # Private variables that shouldn't be carelessly modified
        self._model = model
        self._experiments = experiments  # A list of all the experiments in the project
        self._model_parameter_settings = model_parameter_settings

        # Private variables that are modified depending on experiments in project
        self._project_param_idx = None
        self._n_project_params = None  # How many total parameters are there that are being optimized
        self._residuals_per_param = None  # How many data points do we have to constrain each parameter
        self._n_residuals = None  # How many data points do we have across all experiments
        self._measurements_idx = None  # The index of the experiments where a particular measurement is present
        self._update_project_settings()

        self._measurement_to_model_map = {}
        if set(self._measurements_idx.keys()) != set(measurement_to_model_map.keys()):
            raise KeyError('Measurements without explicit mapping to model variables')
        for measure_name, (mapping_type, mapping_args) in measurement_to_model_map.items():

            if mapping_type == 'direct':
                assert(type(mapping_args) is int)
                parameters = mapping_args  # Index of model variable
                variable_map_fcn = utils.direct_model_var_to_measure
                jacobian_map_fcn = utils.direct_model_jac_to_measure_jac

            elif mapping_type == 'sum':
                assert(type(mapping_args) is list)
                parameters = mapping_args  # Index of model variables
                variable_map_fcn = utils.sum_model_vars_to_measure
                jacobian_map_fcn = utils.sum_model_jac_to_measure_jac

            elif mapping_type == 'custom':
                parameters = mapping_args[0]  # Arbitrary parameters passed to function
                variable_map_fcn = mapping_args[1]
                jacobian_map_fcn = mapping_args[2]

            else:
                raise ValueError('Invalid mapping type')

            mapper = {'parameters': parameters, 'model_variables_to_measure_func': variable_map_fcn,
                      'model_jac_to_measure_jac_func': jacobian_map_fcn}
            self._measurement_to_model_map[measure_name] = mapper

        # Misc Constraints
        self._scale_factors_priors = {}
        self._parameter_priors = {}

        # Variables modified upon simulation:
        self._all_sims = []
        self._all_residuals = None
        self._model_jacobian = None
        self._scale_factors = None
        self._scale_factors_gradient = None
        self.project_param_vector = None

        # Public variables - can modify them to change simulations.
        self.experiments_weights = np.ones((len(experiments),))
        self.use_scale_factors = {measure_name: True for measure_name in self._measurements_idx}
        self.use_parameter_priors = False  # Use the parameter priors in the Jacobian calculation
        self.use_scale_factors_priors = False  # Use the scale factor priors in the Jacobian calculation

    ##########################################################################################################
    # Methods that update private variables
    ##########################################################################################################

    def _update_project_settings(self):
        self._project_param_idx, self._n_project_params, self._residuals_per_param = self._set_local_param_idx()

        # Convenience variables that depend on constructor arguments.
        self._n_residuals = self._set_n_residuals()
        self._measurements_idx = self._set_measurement_idx()

    def _set_local_param_idx(self):
        """
        We map the model parameters to the global parameters using the model parameter settings dictionary.

        Parameters come in 3 flavors:

        - Local: A parameter is local if it is allowed to take on a different value in each experiment
        - Global: A parameter is global if it has a fixed value across all experiments
        - Shared: Shared parameters belong to groups, and each group depends on a number of settings.  All parameters
        in the same group will have the same value in experiments that have the same settings.

        Typically, the number of global parameters will be much bigger than the number of model parameters.

        This function is called when new experiments are added (or removed) from the project.
        """

        all_model_parameters = set(self._model.param_order)
        local_pars = self._model_parameter_settings.get('Local', [])
        project_fixed_pars = self._model_parameter_settings.get('Fixed', [])
        global_pars = self._model_parameter_settings.get('Global', [])

        # Note - shared parameters are grouped together
        shared_pars_groups = self._model_parameter_settings.get('Shared', {})
        shared_pars = set([p for p_group in shared_pars_groups for p in shared_pars_groups[p_group]])

        global_pars.extend(all_model_parameters - set(local_pars) - set(project_fixed_pars) -
                           shared_pars - set(global_pars))
        # Parameters that have no settings are global by default

        _project_param_idx = {}
        # This dict maps the location of parameters in the global parameter vector.
        residuals_per_param = defaultdict(lambda: defaultdict(lambda: 0))

        n_params = 0
        for p in global_pars:
            _project_param_idx[p] = {}
            _project_param_idx[p]['Global'] = n_params
            n_params += 1

        for exp_idx, experiment in enumerate(self._experiments):
            exp_param_idx = OrderedDict()

            try:
                exp_fixed_pars = experiment.fixed_parameters.keys()
            except AttributeError:
                # That experiment has no fixed parameters.
                exp_fixed_pars = []
            all_fixed_pars = project_fixed_pars + exp_fixed_pars

            for p in project_fixed_pars:
                assert(p in exp_fixed_pars)
                # If some parameters are marked as globally fixed, each experiment must provide a value.

            for p in global_pars:
                if p not in all_fixed_pars:
                    exp_param_idx[p] = _project_param_idx[p]['Global']
                    residuals_per_param[p]['Global'] += len(experiment.get_unique_timepoints())
                    # Global parameters always refer to the same index in the global parameter vector

            for p_group in shared_pars_groups:
                # Shared parameters depend on the experimental settings.
                for p, settings in shared_pars_groups[p_group].items():
                    if p in exp_fixed_pars:
                        continue  # Experiment over-writes project settings.

                    settings = tuple(settings)
                    exp_p_settings = tuple([experiment.settings[setting] for setting in settings])
                    # Here we get the experimental conditions for all settings upon which that parameter depends.

                    if p_group not in _project_param_idx:
                        _project_param_idx[p_group] = {}
                    # If it's the first time we encounter that parameter group, create it.

                    try:
                        exp_param_idx[p] = _project_param_idx[p_group][exp_p_settings]
                        # If we already have another parameter in that parameter group and with those same
                        # experimental settings, then they point to the same location.
                    except KeyError:
                        _project_param_idx[p_group][exp_p_settings] = n_params
                        exp_param_idx[p] = n_params
                        n_params += 1
                    residuals_per_param[p_group][exp_p_settings] += len(experiment.get_unique_timepoints())

            for p in local_pars:
                if p in exp_fixed_pars:
                    continue  # Experiment over-writes project settings.
                par_string = '%s_%s' % (p, experiment.name)
                _project_param_idx[par_string]['Local'] = n_params
                exp_param_idx[p] = n_params
                n_params += 1
                residuals_per_param[par_string]['Local'] += len(experiment.get_unique_timepoints())

            self._experiments[exp_idx].param_global_vector_idx = exp_param_idx
        return _project_param_idx, n_params, residuals_per_param

    def _set_measurement_idx(self):
        """
        Lists all experiments that contain measurement of a particular variable
        """
        m_idx = {}
        for i, experiment in enumerate(self._experiments):
            for measurement in experiment.measurements:
                measure_name = measurement.variable_name
                try:
                    m_idx[measure_name].append(i)
                except KeyError:
                    m_idx[measure_name] = []
                    m_idx[measure_name].append(i)
        return m_idx

    def _set_n_residuals(self, include_zero=False):
        """
        Calculates the total number of experimental points across all experiments
        """
        n_res = 0
        for experiment in self._experiments:
            for measurement in experiment.measurements:
                timepoints = measurement.timepoints
                if not include_zero:
                    timepoints = timepoints[timepoints != 0]
                n_res += len(timepoints)
        return n_res

    ##########################################################################################################
    # Private Simulation Methods
    ##########################################################################################################

    def _sim_experiments(self, exp_subset='all', all_timepoints=False):
        """
        Simulates all the experiments in the project.
        """

        if self.project_param_vector is None:
            raise ValueError('Parameter vector not set')
        if exp_subset is 'all':
            simulated_experiments = self._experiments
        else:
            for exp_idx in exp_subset:
                simulated_experiments = self._experiments[exp_idx]

        for experiment in simulated_experiments:
            exp_sim = self._model.simulate_experiment(self.project_param_vector, experiment,
                                                     self._measurement_to_model_map, all_timepoints)
            self._all_sims.append(exp_sim)

    def _calc_scale_factors(self):
        """
        Analytically calculates the optimal scale factor for measurements that are in arbitrary units
        """
        self._scale_factors = {}
        for measure_name, experiment_list in self._measurements_idx.items():
            if self.use_scale_factors[measure_name] is False:
                self._scale_factors[measure_name] = 1
                continue

            sim_dot_exp = np.zeros((1,), dtype='float64')
            sim_dot_sim = np.zeros((1,), dtype='float64')

            for exp_idx in experiment_list:
                experiment = self._experiments[exp_idx]
                exp_weight = self.experiments_weights[exp_idx]

                measurement = experiment.get_variable_measurements(measure_name)
                exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()

                sim_data = self._all_sims[exp_idx][measure_name]['value']
                _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, exp_weight)

            self._scale_factors[measure_name] = sim_dot_exp / sim_dot_sim

    def _calc_scale_factors_gradient(self):
        """
        Analytically calculates the jacobian of the scale factors for each measurement
        """
        self._scale_factors_gradient = {}
        n_global_pars = len(self.project_param_vector)
        for measure_name, experiment_list in self._measurements_idx.items():
            scale_factor_gradient = np.zeros((n_global_pars,), dtype='float64')

            if self.use_scale_factors[measure_name]:
                sens_dot_exp_data = np.zeros((n_global_pars,), dtype='float64')
                sens_dot_sim = np.zeros((n_global_pars,), dtype='float64')
                sim_dot_sim = np.zeros((1,), dtype='float64')
                sim_dot_exp = np.zeros((1,), dtype='float64')

                for exp_idx in experiment_list:
                    experiment = self._experiments[exp_idx]
                    exp_weight = self.experiments_weights[exp_idx]

                    measurement = experiment.get_variable_measurements(measure_name)
                    exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()

                    sim_data = self._all_sims[exp_idx][measure_name]['value']  # Vector
                    model_sens = self._model_jacobian[exp_idx][measure_name]  # Matrix

                    _accumulate__scale_factors_jac(exp_data, exp_std, sim_data, model_sens, sim_dot_exp, sim_dot_sim,
                                                   sens_dot_exp_data, sens_dot_sim, exp_weight)

                _combine__scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp,
                                        scale_factor_gradient)

            self._scale_factors_gradient[measure_name] = scale_factor_gradient

    def _calc_scale_factor_prior_gradient(self, measure_name, scale_factor_gradient):
        scale_factor = self.scale_factors[measure_name]
        log_scale_factor = np.log(scale_factor)
        log_scale_factor_prior, log_sigma_scale_factor = self._scale_factors_priors[measure_name]
        scale_factor_prior_penalty = ((scale_factor_gradient / log_sigma_scale_factor ** 2) *
                                      (log_scale_factor - log_scale_factor_prior))
        # scale factor log priors are of form: (log(B_i(theta)) - log(B_*) / sigma_log_B_*)**2
        # derive that wrt theta and you get above formula
        return scale_factor_prior_penalty

    def _calc_parameters_prior_residuals(self):
        parameter_residuals = []
        for parameter_group in self._parameter_priors:
            for setting in self._parameter_priors[parameter_group]:
                p_idx = self._project_param_idx[parameter_group][setting]
                log_p_value = self.project_param_vector[p_idx]
                log_scale_parameter_prior, log_sigma_parameter = self._parameter_priors[parameter_group][setting]
                res = log_p_value - log_scale_parameter_prior / log_sigma_parameter
                parameter_residuals.append((p_idx, res))

        parameter_residuals.sort(lambda x: x[0])
        return zip(*parameter_residuals)[0]

    def _calc_parameters_prior_jacobian(self):
        parameter_priors = []
        for parameter_group in self._parameter_priors:
            for setting in self._parameter_priors[parameter_group]:
                p_idx = self._project_param_idx[parameter_group][setting]
                log_p_value = self.project_param_vector[p_idx]
                exp_inv = 1 / np.exp(log_p_value)
                parameter_priors.append((p_idx, exp_inv))

        parameter_prior_jac = np.zeros((len(parameter_priors, self.n_project_params)))
        for p_idx, exp_inv in parameter_priors:
            parameter_prior_jac[p_idx, p_idx] = exp_inv

        return parameter_prior_jac

    def _calc_scale_factor_entropy(self, measure_name, temperature):
        """
        Implementation taken from SloppyCell.  All credit to Sethna group, all mistakes are mine
        """
        if measure_name not in self._scale_factors_priors:
            return 0

        log_scale_factor_prior, log_sigma_scale_factor = self._scale_factors_priors[measure_name]
        sim_dot_exp = np.zeros((1,), dtype='float64')
        sim_dot_sim = np.zeros((1,), dtype='float64')

        for exp_idx in self._measurements_idx[measure_name]:
            experiment = self._experiments[exp_idx]
            exp_weight = self.experiments_weights[exp_idx]
            measurement = experiment.get_variable_measurements(measure_name)
            exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()

            sim_data = self._all_sims[exp_idx][measure_name]['value']
            _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, exp_weight)

        optimal_scale_factor = sim_dot_exp / sim_dot_sim
        log_optimal_scale_factor = np.log(optimal_scale_factor)

        integral_args = (sim_dot_sim, sim_dot_exp, log_scale_factor_prior, log_sigma_scale_factor, temperature,
                         optimal_scale_factor, log_optimal_scale_factor)
        ans, temp = scipy.integrate.quad(_entropy_integrand, -scipy.inf, scipy.inf, args=integral_args, limit=1000)

        entropy = np.log(ans)
        return entropy

    def _calc_exp_residuals(self, exp_idx):
        """Returns the residuals between simulations (after scaling) and experimental measurents.
        """
        residuals = OrderedDict()
        experiment = self._experiments[exp_idx]

        exp_weight = self.experiments_weights[exp_idx]
        for measurement in experiment.measurements:
            measure_name = measurement.variable_name
            scale = self._scale_factors[measure_name]

            exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()
            sim_data = self._all_sims[exp_idx][measure_name]['value']

            residuals[measure_name] = (sim_data * scale - exp_data) / exp_std
        return residuals

    def _calc_residuals(self):
        """
        Calculates residuals between the simulations, scaled by the scaling factor and the simulations.
        Has to be called after simulations and scaling factors are calculated
        """
        _all_residuals = []
        for exp_idx, experiment in enumerate(self._experiments):
            exp_res = self._calc_exp_residuals(exp_idx)
            _all_residuals.append(exp_res)
        self._all_residuals = _all_residuals

    def _calc_model_jacobian(self):
        all_jacobians = []
        for experiment in self._experiments:
            if self.project_param_vector is None:
                raise ValueError('Parameter vector not set')
            exp_jacobian = self._model.calc_jacobian(self.project_param_vector,
                                                    experiment, self._measurement_to_model_map)
            all_jacobians.append(exp_jacobian)
        self._model_jacobian = all_jacobians

    ##########################################################################################################
    # Public API
    ##########################################################################################################

    @property
    def n_project_params(self):
        return self._n_project_params

    @property
    def scale_factors(self):
        return copy.deepcopy(self._scale_factors)

    @property
    def experiments(self):
        return iter(self._experiments)

    def get_experiment(self, exp_idx):
        return copy.deepcopy(self._experiments[exp_idx])

    def add_experiment(self, experiment):
        """
        Adds an experiment to the project

        Attributes
        ----------
        experiment: :class:`~OdeModel:experiment.experiments.Experiment`
            An experiment
        """
        self._experiments.append(experiment)
        self.experiments_weights = np.append(self.experiments_weights, 1)

        # Now we update the project parameter settings.
        self._update_project_settings()

    def set_scale_factor_log_prior(self, measure_name, log_scale_factor_prior, log_sigma_scale_factor):
        if measure_name not in self._measurement_to_model_map:
            raise KeyError('%s not in project measures' % measure_name)
        if self.use_scale_factors[measure_name] is False:
            raise ValueError("Cannot set priors on a scale factor we are not calculating")

        self._scale_factors_priors[measure_name] = (log_scale_factor_prior, log_sigma_scale_factor)

    def set_parameter_log_prior(self, parameter, parameter_setting, log_scale_parameter_prior, log_sigma_parameter):
        try:
            self._project_param_idx[parameter][parameter_setting]
        except KeyError:
            raise KeyError('%s with settings %s not in the project parameters')
        self._parameter_priors[parameter][parameter_setting] = (log_scale_parameter_prior, log_sigma_parameter)

    def remove_experiment(self, experiment_name):
        removed_exp_idx = -1
        for exp_idx, experiment in enumerate(self._experiments):
            if experiment.name == experiment_name:
                removed_exp_idx = exp_idx
                break

        if removed_exp_idx == -1:
            raise KeyError('%s is not in the list of experiments' % experiment_name)
        else:
            self.experiments_weights = np.delete(self.experiments_weights, removed_exp_idx)
            self._experiments.pop(removed_exp_idx)
            self._update_project_settings()

    def remove_experiments_by_settings(self, list_of_settings):
        removed_exp_idx = []

        for exp_idx, experiment in enumerate(self._experiments):
            for settings in list_of_settings:
                all_equal = 1
                for setting in settings:
                    if setting not in experiment.settings:
                        raise KeyError('%s is not a setting in experiment %s' % (setting, experiment.name))
                    if not (experiment.settings[setting] == settings[setting]):
                        all_equal = 0
                if all_equal and exp_idx not in removed_exp_idx:
                    removed_exp_idx.append(exp_idx)

        if len(removed_exp_idx) == 0:
            raise KeyError('None of the settings chosen were in the experiments in the project')

        del_indices = np.ones((len(self.experiments_weights),))
        for exp_idx in removed_exp_idx:
            del_indices[exp_idx] = False

        # Have to traverse in reverse order to remove highest idx experiments first
        for exp_idx in reversed(removed_exp_idx):
            self.experiments_weights = np.delete(self.experiments_weights, exp_idx)
            self._experiments.pop(exp_idx)
        self._update_project_settings()

        if len(self._experiments) == 0:
            warnings.warn('Project has no more experiments')

    def get_experiment_index(self, exp_name):
        for exp_idx, experiment in enumerate(self._experiments):
            if exp_name == experiment.name:
                return exp_idx
        raise KeyError('%s not in experiments')

    def update_experimental_weights(self, new_weights):
        for exp_name, exp_weight in new_weights:
            exp_idx = self.get_experiment_index(exp_name)
            self.experiments_weights[exp_idx] = exp_weight

    def reset_calcs(self):
        self._all_sims = []
        self._all_residuals = None
        self._model_jacobian = None
        self._scale_factors = None
        self._scale_factors_gradient = None
        self.project_param_vector = None

    def print_param_settings(self, verbose=True):
        """
        Prints out all the parameters combinations in the project in a fancy way
        """

        total_params = 0
        for p_group in self._project_param_idx:
            exp_settings = self._project_param_idx[p_group].keys()
            exp_settings = sorted(exp_settings)
            if verbose:
                print '%s  total_settings: %d ' % (p_group, len(exp_settings))
            for exp_set in exp_settings:
                if verbose:
                    print '%s, %d \t' % (repr(exp_set), self._residuals_per_param[p_group][exp_set])
                total_params += 1
            if verbose:
                print '\n***********************'
        if verbose:
            print "Total Parameters: %d" % total_params
        return total_params

    def get_ordered_project_params(self):
        project_params = []
        for p_group in self._project_param_idx:
            exp_settings = self._project_param_idx[p_group].keys()
            for exp_set in exp_settings:
                global_idx = self._project_param_idx[p_group][exp_set]
                global_p_name = p_group + ' ' + ''.join([str(setting) for setting in exp_set])
                project_params.append((global_p_name, global_idx))
        project_params.sort(key=lambda x: x[1])
        return zip(*project_params)[0]

    def __call__(self, project_param_vector):
        """
        Calculates the residuals between the simulated values (after optimal scaling) and the experimental values
        across all experiments in the project.  Note about order.

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        residual_array: :class:`~numpy:numpy.ndarray`
            An (m,) dimensional array where m is the number of residuals
        """
        if (self.project_param_vector is None) or (not np.alltrue(project_param_vector == self.project_param_vector)):
            self.reset_calcs()
            self.project_param_vector = np.copy(project_param_vector)

            self._sim_experiments()
            self._calc_scale_factors()
            self._calc_residuals()

        residual_array = np.zeros((self._n_residuals,))
        res_idx = 0
        for exp_res in self._all_residuals:
            for res_block in exp_res.values():
                residual_array[res_idx:res_idx+len(res_block)] = res_block
                res_idx += len(res_block)

        return residual_array

    def calc_project_jacobian(self, project_param_vector):
        """
        Given a cost function:

        .. math::
            C(\\theta)= 0.5*(\\sum{BY(\\theta)_{sim} - Y_{exp}})^2

        Where:
         :math:`B` are the optimal scaling factors for each measure \n
         :math:`\\theta` are the parameters that we are optimizing \n
         :math:`Y(\\theta)_{sim}` is the output of the model as a function of the parameters \n
         :math:`Y_{exp}` is the experimental data \n

        The global jacobian is:

        .. math::
            &= \\frac{\\partial BY_{sim}}{\\partial \\theta} \\
            &= B\\frac{\\partial Y_{sim}}{\\partial \\theta} + Y_{sim}\\frac{\\partial B}{\\partial \\theta}

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        include_scale_factors: bool, optional
            If true, we include the derivative terms with respect to B.  If false, we only consider
            the jacobian with respect to the model (useful for calculating Hessian).

        Returns
        -------
        calc_project_jacobian: :class:`~numpy:numpy.ndarray`
            An (n, m) array where m is the number of residuals and n is the number of global parameters.
        """
        if self.project_param_vector is None or not np.alltrue(project_param_vector == self.project_param_vector):
            self.reset_calcs()
            self.project_param_vector = np.copy(project_param_vector)

            self._sim_experiments()
            self._calc_scale_factors()
            self._calc_residuals()

            self._calc_model_jacobian()
            self._calc_scale_factors_gradient()

        elif self._model_jacobian is None:
            self._calc_model_jacobian()
            self._calc_scale_factors_gradient()

        elif self._scale_factors_gradient is None:
            self._calc_scale_factors_gradient()

        jacobian_array = np.zeros((self._n_residuals, len(project_param_vector)))
        res_idx = 0
        for exp_jac, exp_sim in zip(self._model_jacobian, self._all_sims):
            for measure_name in exp_jac:
                measure_sim = exp_sim[measure_name]['value']
                measure_sim_jac = exp_jac[measure_name]
                measure_scale = self._scale_factors[measure_name]
                measure_scale_grad = self._scale_factors_gradient[measure_name]
                if self.use_scale_factors[measure_name]:
                    jac = measure_sim_jac*measure_scale + measure_scale_grad.T*measure_sim[:, np.newaxis]
                    # J = dY_sim/dtheta * B + dB/dtheta * Y_sim
                else:
                    jac = measure_sim_jac
                jacobian_array[res_idx:res_idx+len(measure_sim), :] = jac
                res_idx += len(measure_sim)

        return jacobian_array

    def calc_rss_gradient(self, project_param_vector):
        """
        Returns the gradient of the cost function (the residual sum of squares).

        The cost function is:

        .. math::
            C(\\theta)= 0.5*(\\sum{BY(\\theta)_{sim} - Y_{exp}})^2

        This function returns

        .. math::
            \\frac{\\partial C}{\\partial \\theta}

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        calc_project_jacobian: :class:`~numpy:numpy.ndarray`
            An (m,) dimensional array.

        See Also
        --------
        calc_project_jacobian
        """
        jacobian_matrix = self.calc_project_jacobian(project_param_vector)
        residuals = self(project_param_vector)
        return (jacobian_matrix.T * residuals).sum(axis=1)

    def calc_sum_square_residuals(self, project_param_vector):
        """
        Returns the sum squared residuals across all experiments

        Returns:

        .. math::
            C(\\theta)= 0.5*(\\sum{BY_{sim} - Y_{exp}})^2

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        rss: : float
            The sum of all residuals
        """
        residuals = self(project_param_vector)
        return 0.5*np.sum(residuals**2)

    def nlopt_fcn(self, project_param_vector, grad):
        """
        Wrapper to make objective function compatible with nlopt API

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        grad: :class:`~numpy:numpy.ndarray`, optional
            Cost function gradient.  Only used for gradient based optimization by nlopt

        Returns
        -------
        rss : float
            The sum of all residuals

        See Also
        --------
        calc_sum_square_residuals
        """
        if grad.size > 0:
            grad[:] = self.calc_rss_gradient(project_param_vector)
        return self.calc_sum_square_residuals(project_param_vector)

    def load_param_dict(self, param_dict, default_value=0.0):
        """
        Loads parameters from a dictionary (default output format for saving at the end of a fit)

        Parameters
        ----------
        param_dict: dict
            Dictionary of parameters

        default_value: float
            Value for the project_parameter_vector in case it's not present in the param_dict

        Returns
        -------
        param_vector: :class:`~numpy:numpy.ndarray`
            An array with the parameter values of param_dict in the correct index.

        See Also
        --------
        param_vect_to_dict
        """
        param_vector = np.ones((self.n_project_params,)) * default_value
        for p_group in self._project_param_idx:
            for exp_settings, global_idx in self._project_param_idx[p_group].items():
                try:
                    param_vector[global_idx] = param_dict[p_group][exp_settings]
                except KeyError:
                    pass
        return param_vector

    def project_param_vect_to_dict(self, param_vector):
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
        for p_group in self._project_param_idx:
            param_dict[p_group] = {}
            for exp_settings, global_idx in self._project_param_idx[p_group].items():
                param_dict[p_group][exp_settings] = param_vector[global_idx]
        return param_dict

    def model_hessian(self, project_param_vector):
        pass

    def parameter_uncertainty(self, project_param_vector):
        pass

    def get_total_scale_factor_entropy(self, temperature=1.0):
        """
        Calculates the entropy from the scale factors.  Currently only log priors are supported.

        This term is useful when doing sampling (and optimization) to keep the sampling function
        in a narrow range near the minima.

        Parameters
        ----------
        temperature: float, optional
            The temperature for the evaluation of the entropy

        Returns
        -------
        param_vector: float
            The sum of the entropy of all scale factors in the model
        """
        entropy = 0
        for measure_name in self._measurement_to_model_map:
            if self.use_scale_factors[measure_name]:
                entropy += temperature * self._calc_scale_factor_entropy(measure_name, temperature)
        return entropy

