from collections import OrderedDict, defaultdict
import warnings
import copy

import numpy as np
import pandas as pd

from . import utils
from loss_functions.squared_loss import SquareLossFunction


class Project(object):
    """Class to simulate experiments with a given model

    Attributes
    ----------
    model : :class:`~OdeModel:SysBio_Modeling.model.ode_model.OdeModel`
        A model that can simulate experiments
    experiments: list
        A collection of experiments.
    model_parameter_settings : dict
        How the parameters in the model vary depending on the experimental settings in the experiments
    measurement_to_model_map : dict
        A dictionary of functions that maps the variables simulated by the model to the observed
         measurements in the experiments.
    loss_function: string, optional, d
        How to scale simulations to map them to measurements.
    sf_groups: list
        A list of tuples of measurements that have to share the same scale factor
    """

    def __init__(self, model, experiments, model_parameter_settings, measurement_to_model_map,
                 sf_groups=None):
        """


        :type self: project.base_project.Project
        :param model: An instance of the Model class that simulates experiments
        :type model: model.ode_model.OdeModel
        :param experiments: A list of Experiment objects
        :type experiments: list[experiment.experiments.Experiment]
        :param model_parameter_settings: A dictionary which specifies how each parameter in the model is affected
            by the Experiment settings.
        :type model_parameter_settings: OrderedDict
        :param measurement_to_model_map: A struct that contains the necessary information to map the output of the model
        to the measurements of the model
        # TODO: Refactor as a class.
        :type measurement_to_model_map: dict
        :param sf_groups: Which measurements share scale factors
        :type sf_groups: list[set]
        :return:
        :rtype: None
        """
        self.project_description = ""

        # Private variables that shouldn't be carelessly modified
        ###############################################################################################################
        self._model = model
        self._model_parameter_settings = model_parameter_settings
        self._loss_function = SquareLossFunction(sf_groups)

        # Take care of experiments now
        self._experiments = []  # A list of all the experiments in the project
        self.add_experiment(experiments)  # We use this to add the experiments to insure they are lex-sorted

        self._parameter_priors = OrderedDict()
        # Priors on Parameter
        ###############################################################################################################

        # Private variables that are modified depending on experiments in project
        ###############################################################################################################
        self._project_param_idx = None
        self._n_project_params = None  # How many total parameters are there that are being optimized
        self._residuals_per_param = None  # How many data points do we have to constrain each parameter
        self._n_residuals = None  # How many data points do we have across all experiments
        self._measurements_idx = None  # The index of the experiments where a particular measurement is present
        self._measurement_df = None
        self._simulations_df = None
        self._update_project_settings()  # This initializes all the above variables

        self._measurement_to_model_map = {}
        if set(self._measurements_idx.keys()) != set(measurement_to_model_map.keys()):
            raise KeyError('Measurements without explicit mapping to model variables')
        for measure_name, (mapping_type, mapping_args) in measurement_to_model_map.items():

            if mapping_type == 'direct':
                assert(type(mapping_args) is int)
                if mapping_args >= model.n_vars:
                    raise ValueError('Index (%d) has to be smaller than %d' % (mapping_args, model.n_vars))
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

        # Variables modified upon simulation:
        ###############################################################################################################
        self._model_jacobian = None
        self._project_param_vector = np.zeros((self.n_project_params,))

        # Public variables - can modify them to change simulations.

        self.use_parameter_priors = False  # Use the parameter priors in the Jacobian calculation

    ##########################################################################################################
    # Methods that update private variables
    ##########################################################################################################

    def _update_project_settings(self):

        self._project_param_idx, self._n_project_params, self._residuals_per_param = self._set_local_param_idx()

        # Convenience variables that depend on constructor arguments.
        self._n_residuals = self._set_n_residuals()
        self._measurements_idx = self._set_measurement_idx()
        self._measurement_df = self._measurements_as_dataframe()

        self._simulations_df = self._measurement_df.copy()
        self._simulations_df.drop('std', axis=1, inplace=True)
        self._simulations_df.values[:] = 0

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

        no_settings_params = (all_model_parameters - set(local_pars) - set(project_fixed_pars) -
                              shared_pars - set(global_pars))

        if len(no_settings_params) > 0:
            par_str = ", ".join(no_settings_params)
            print "The following parameters are global because no settings were specified: %s " % par_str

        global_pars.extend(no_settings_params)
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
                try:
                    assert(p in exp_fixed_pars)
                    # If some parameters are marked as globally fixed, each experiment must provide a value.
                except AssertionError:
                    raise ValueError('%s was declared as a fixed parameter, but in experiment %s no value provided'
                                     % (p, experiment.name))

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

                    if settings is None:
                        exp_p_settings = 'None'
                        # Shared Global parameter group
                    else:
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
        m_idx = OrderedDict()
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
    def _measurements_as_dataframe(self):
        measurement_df = np.zeros((3, 0))
        df_index = []
        for experiment in self._experiments:
            for measurement in experiment.measurements:
                vals = measurement.get_nonzero_measurements()
                vals = np.array(vals)
                measurement_df = np.hstack((measurement_df, vals))

                for i in range(vals.shape[1]):
                    df_index.append((experiment.name, measurement.variable_name))

        df_index = pd.MultiIndex.from_tuples(df_index)
        measurement_df = pd.DataFrame(np.array(measurement_df).T, index=df_index, columns=['mean', 'std',
                                                                                           'timepoints'])
        measurement_df.sortlevel(inplace=True)
        return measurement_df

    def _get_experiment_parameters(self, experiment):
        """
        Extracts the experiment-specific parameters from the project parameter vector
        """
        m = self._model
        exp_param_vector = np.zeros((len(m.param_order),))
        for p_model_idx, p_name in enumerate(m.param_order):
            try:
                global_idx = experiment.param_global_vector_idx[p_name]
                param_value = self._project_param_vector[global_idx]
                param_value = np.exp(param_value)
            except KeyError:
                param_value = experiment.fixed_parameters[p_name]
                # If it's not an optimized parameter, it must be fixed by experiment
            exp_param_vector[p_model_idx] = param_value
        return exp_param_vector

    def _map_model_sim_to_measures(self, model_sim, t_sim, experiment, res_idx, use_experimental_timepoints=True):
        """
        Maps a model simulation to a particular measurement.  This is necessary because not all model variables
        map cleanly to a single measurement.
        """
        measure_sim_dict = OrderedDict()

        for measurement in experiment.measurements:
            measure_name = measurement.variable_name
            mapping_struct = self._measurement_to_model_map[measure_name]
            model_variables_to_measure_func = mapping_struct['model_variables_to_measure_func']
            mapping_parameters = mapping_struct['parameters']

            measure_sim_dict[measure_name] = {}
            mapped_sim, mapped_timepoints = model_variables_to_measure_func(model_sim, t_sim, experiment, measurement,
                                                                            mapping_parameters,
                                                                            use_experimental_timepoints)
            extra_residuals = len(mapped_sim)
            temp = np.array((mapped_sim, mapped_timepoints)).T
            self._simulations_df.values[res_idx:res_idx+extra_residuals, :] = temp

            #self._simulations_df.ix[(experiment.name, measure_name)].values = temp
            #self._simulations_df.loc[(measure_name, experiment.name), 'values'] = mapped_sim
            #self._simulations_df.loc[(measure_name, experiment.name), 'timepoints'] = mapped_timepoints
            extra_residuals = len(mapped_sim)
        return extra_residuals

    def _sim_experiments(self, exp_subset='all', use_experimental_timepoints=True):
        """
        Simulates all the experiments in the project.

        use_experimental_timepoints should always be true when the simulations are used to calculate
        scale factors.
        """

        if self._project_param_vector is None:
            raise ValueError('Parameter vector not set')

        simulated_experiments = []

        if exp_subset is 'all':
            simulated_experiments = self._experiments
        else:
            for exp_idx in exp_subset:
                simulated_experiments.append(self._experiments[exp_idx])

        residual_idx = 0
        for experiment in simulated_experiments:
            experiment_parameters = self._get_experiment_parameters(experiment)
            t_end = experiment.get_unique_timepoints()[-1]
            t_sim = np.linspace(0, t_end, 1000)
            model_sim = self._model.simulate(experiment_parameters, t_sim)
            residual_idx += self._map_model_sim_to_measures(model_sim, t_sim, experiment,
                                                            residual_idx, use_experimental_timepoints)
    ##########################################################################################################
    # Sensitivity Methods
    ##########################################################################################################

    def _calc_model_jacobian(self):
        all_jacobians = []
        n_vars = self._model.n_vars

        for experiment in self._experiments:
            if self._project_param_vector is None:
                raise ValueError('Parameter vector not set')

            experiment_parameters = self._get_experiment_parameters(experiment)
            t_end = experiment.get_unique_timepoints()[-1]
            t_sim = np.linspace(0, t_end, 1000)
            n_nonfixed_experiment_params = len(experiment.param_global_vector_idx)
            init_conditions = np.zeros((n_vars + n_nonfixed_experiment_params * n_vars,))

            model_jacobian = self._model.calc_jacobian(experiment_parameters, t_sim, init_conditions)
            project_jacobian_dict = self._model_jacobian_to_project_jacobian(model_jacobian, t_sim, experiment)
            all_jacobians.append(project_jacobian_dict)

        self._model_jacobian = all_jacobians

    def _model_jacobian_to_project_jacobian(self, jacobian_sim, t_sim, experiment):
        """
        Map the jacobian with respect to model parameters to the jacobian with respect to the global parameters
        """
        m = self._model

        transformed_params_deriv = np.exp(self._project_param_vector)
        # The project parameters are in log space, so:
        # f(g(x)) where g(x) is e^x - so d(f(g(x))/dx = df/dx(g(x))*dg/dx(x)
        # dg/dx = e^x

        jacobian_dict = OrderedDict()
        for measurement in experiment.measurements:
            measure_name = measurement.variable_name

            # We convert the model state jacobian to measure variables
            mapping_struct = self._measurement_to_model_map[measure_name]
            model_jac_to_measure_func = mapping_struct['model_jac_to_measure_jac_func']
            mapping_parameters = mapping_struct['parameters']
            measure_jac = model_jac_to_measure_func(jacobian_sim, t_sim, experiment, measurement, mapping_parameters)

            var_jacobian = np.zeros((measure_jac.shape[0], len(self._project_param_vector)))

            for p_model_idx, p_name in enumerate(m.param_order):
                # p_model_idx is the index of the parameter in the model
                # p_name is the name of the parameter
                try:
                    p_project_idx = experiment.param_global_vector_idx[p_name]
                    # p_project_idx is the index of a parameter in the project vector
                except KeyError:
                    if p_name not in experiment.fixed_parameters:
                        raise KeyError('%s not in %s fixed parameters.' % (p_name, experiment.name))
                    else:
                        continue
                        # We don't calculate the jacobian wrt fixed parameters.
                var_jacobian[:, p_project_idx] += measure_jac[:, p_model_idx]
            jacobian_dict[measurement.variable_name] = var_jacobian * transformed_params_deriv

        return jacobian_dict

    ##########################################################################################################
    # Scale Factor Methods
    ##########################################################################################################

    def _update_scale_factors(self):
        """
        Analytically calculates the optimal scale factor for measurements that are in arbitrary units
        """
        for measure_name in self._scale_factors:


                sf_iter = self.measure_iterator(measure_name)
                self._scale_factors[measure_name].update_sf(sf_iter)

    def _update_scale_factors_gradient(self):
        """
        Analytically calculates the jacobian of the scale factors for each measurement
        """
        for measure_name in self._scale_factors:
            if self.use_scale_factors[measure_name]:
                sf_iter = self.measure_iterator(measure_name)
                self._scale_factors[measure_name].update_sf_gradient(sf_iter, self.n_project_params)

    def _calc_scale_factors_prior_jacobian(self):
        """
        prior penalty is: ((log(B(theta)) - log_B_prior) / sigma_b_prior)**2

        derive (log(B(theta)) = 1/B(theta) * dB/dtheta
        dB/dtheta is the scale factor gradient
        """
        scale_factor_priors_jacobian = []
        for measure_name in self._scale_factors:
            grad = self._scale_factors[measure_name].calc_sf_prior_gradient()
            if grad is not None:
                scale_factor_priors_jacobian.append(grad)
        return np.array(scale_factor_priors_jacobian)

    def _calc_scale_factors_prior_residuals(self):
        sf_residuals = []
        for measure_name in self._scale_factors:
            res = self._scale_factors[measure_name].calc_sf_prior_residual()
            if res is not None:
                sf_residuals.append(res)
        return np.array(sf_residuals)

    ##########################################################################################################
    # Parameter Priors
    ##########################################################################################################

    def _calc_parameters_prior_jacobian(self):
        """
        Since all parameters are already in logspace, and the priors are all in log space:
        parameter_prior = (log(theta) - log_prior / sigma) **2 -
        We work with log(theta) directly, so d(log(theta))/d(log(theta)) is 1.
        """
        parameter_priors_jacobian = []
        for parameter_group in self._parameter_priors:
            for setting in self._parameter_priors[parameter_group]:
                p_idx = self._project_param_idx[parameter_group][setting]
                #log_p_value = self._project_param_vector[p_idx]
                #exp_inv = 1 / np.exp(log_p_value)
                jac = np.zeros((self.n_project_params,))
                jac[p_idx] = 1.0
                parameter_prioqrs_jacobian.append(jac)

        return np.array(parameter_priors_jacobian)

    def _calc_parameters_prior_residuals(self):
        """
        Due to internal use of OrderedDict for _parameter_priors and _parameter_priors[parameter_group] the
        order in which the residuals and the jacobian are calculated is the same
        """
        parameter_prior_residuals = []

        for parameter_group in self._parameter_priors:
            for setting in self._parameter_priors[parameter_group]:
                p_idx = self._project_param_idx[parameter_group][setting]
                log_p_value = self._project_param_vector[p_idx]
                log_parameter_prior, log_sigma_parameter = self._parameter_priors[parameter_group][setting]
                res = (log_p_value - log_parameter_prior) / log_sigma_parameter
                parameter_prior_residuals.append(res)

        return np.array(parameter_prior_residuals)

    ##########################################################################################################
    # Setters and Getters
    ##########################################################################################################

    @property
    def n_project_params(self):
        return self._n_project_params

    @property
    def n_project_residuals(self):
        return self._n_residuals

    @property
    def scale_factors(self):
        return copy.deepcopy(self._scale_factors)

    @property
    def experiments(self):
        return iter(self._experiments)

    @property
    def project_param_vector(self):
        return np.copy(self._project_param_vector)

    def add_experiment(self, experiment):
        """
        Adds an experiment to the project

        Attributes
        ----------
        experiment: :class:`~OdeModel:experiment.experiments.Experiment`
            An experiment
        """
        if type(experiment) is not list:
            experiment = [experiment]

        experiment_names = {e.name for e in self._experiments}
        # Check that an experiment with that name is not already present:
        for e in experiment:
            if e.name not in experiment_names:
                self._experiments.append(e)
                experiment_names.add(e.name)
            else:
                raise KeyError("An Experiment with name %s is already present" % e.name)

        self._experiments.sort(key=lambda x: x.name)  # Sort experiments by their name

        # Now we update the project parameter settings.
        self._update_project_settings()

    def remove_experiments_by_settings(self, settings):
        """
        Removes all experiments with the specified settings

        Parameters
        ----------
        settings: dict
            A dictonary of {setting_name: setting}.  All experiments for which all setting match are removed
        """

        # Iterate through experiments and find all matches
        removed_exp_idx = []
        for exp_idx, experiment in enumerate(self._experiments):
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

        # Have to traverse in reverse order to remove highest idx experiments first
        deleted_experiments = []
        for exp_idx in reversed(removed_exp_idx):
            deleted_experiments.append(self._experiments.pop(exp_idx))

        self._update_project_settings()

        if len(self._experiments) == 0:
            warnings.warn('Project has no more experiments')

        return deleted_experiments

    def measure_group_iterator(self, measure_name):
        if type(measure_name) is str:
            measure_group = [measure_name]

        else:
            measure_group = list(measure_name)
            measure_group.sort()

        for measure_name in measure_group:
            yield measure_name

    def get_experiment(self, exp_idx):
        return copy.deepcopy(self._experiments[exp_idx])

    def get_experiment_index(self, exp_name):
        for exp_idx, experiment in enumerate(self._experiments):
            if exp_name == experiment.name:
                return exp_idx
        raise KeyError('%s not in experiments')

    def set_scale_factor_log_prior(self, measure_name, log_scale_factor_prior, log_sigma_scale_factor):
        if measure_name not in self._measurement_to_model_map:
            if measure_name not in self._scale_factors:
            # Maybe it's a scale factor group
                raise KeyError('%s not in project measures' % measure_name)
        if self.use_scale_factors[measure_name] is False:
            raise ValueError("Cannot set priors on a scale factor we are not calculating")

        self._scale_factors[measure_name].log_prior = log_scale_factor_prior
        self._scale_factors[measure_name].log_prior_sigma = log_sigma_scale_factor

    def get_scale_factor_priors(self):
        sf_priors = {}
        for measure_name in self._scale_factors:
            sf_priors[measure_name] = (self._scale_factors[measure_name].log_prior,
                                       self._scale_factors[measure_name].log_prior_sigma)
        return sf_priors

    def set_parameter_log_prior(self, parameter, parameter_setting, log_scale_parameter_prior, log_sigma_parameter):
        try:
            self._project_param_idx[parameter][parameter_setting]
        except KeyError:
            raise KeyError('%s with settings %s not in the project parameters' % (parameter, parameter_setting))
        if parameter not in self._parameter_priors:
            self._parameter_priors[parameter] = OrderedDict()
            """This is an OrderedDict within an OrderedDict.  Iteration order is guaranteed to be stable"""

        self._parameter_priors[parameter][parameter_setting] = (log_scale_parameter_prior, log_sigma_parameter)

    def get_parameter_priors(self):
        return copy.deepcopy(self._parameter_priors)

    def get_param_index(self, parameter_group, settings='all'):
        if settings is 'all':
            return self._project_param_idx[parameter_group]
        else:
            return self._project_param_idx[parameter_group][settings]

    def reset_calcs(self):
        self._all_sims = []
        self._all_residuals = None
        self._model_jacobian = None
        self._project_param_vector = np.zeros((self.n_project_params,))

    ##########################################################################################################
    # Public Simulation Methods
    ##########################################################################################################

    def residuals(self, project_param_vector):
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

        self.reset_calcs()
        self._project_param_vector = np.copy(project_param_vector)
        self._sim_experiments()

        return self._loss_function(self._simulations_df, self._measurement_df)

        #measurement_residuals = np.zeros((self._n_residuals,))
        #res_idx = 0
        #for exp_res in self._all_residuals:
        #    for res_block in exp_res.values():
        #        measurement_residuals[res_idx:res_idx + len(res_block)] = res_block
        #        res_idx += len(res_block)

        #project_residuals = measurement_residuals
        #if self.use_parameter_priors and len(self._parameter_priors):
        #    parameter_priors_residuals = self._calc_parameters_prior_residuals()
        #    project_residuals = np.hstack((project_residuals, parameter_priors_residuals))

        #if self.use_scale_factors_priors and len(self.scale_factors):
        #    scale_factor_priors_residuals = self._calc_scale_factors_prior_residuals()
        #    project_residuals = np.hstack((project_residuals, scale_factor_priors_residuals.ravel()))

        #return project_residuals

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
        self.reset_calcs()
        self._project_param_vector = np.copy(project_param_vector)

        self._sim_experiments()
        self._update_scale_factors()
        self._calc_all_residuals()

        self._calc_model_jacobian()
        self._update_scale_factors_gradient()

        measurements_jacobian = np.zeros((self._n_residuals, len(project_param_vector)))
        res_idx = 0
        for exp_jac, exp_sim in zip(self._model_jacobian, self._all_sims):
            for measure_name in exp_jac:
                measure_sim = exp_sim[measure_name]['value']
                measure_sim_jac = exp_jac[measure_name]
                if self.use_scale_factors[measure_name]:
                    sf = self._scale_factors[measure_name].sf
                    sf_grad = self._scale_factors[measure_name].gradient
                    jac = measure_sim_jac * sf + sf_grad.T * measure_sim[:, np.newaxis]
                    # J = dY_sim/dtheta * B + dB/dtheta * Y_sim
                else:
                    jac = measure_sim_jac
                measurements_jacobian[res_idx:res_idx + len(measure_sim), :] = jac
                res_idx += len(measure_sim)

        project_jacobian = measurements_jacobian
        if self.use_parameter_priors and len(self._parameter_priors):
            parameter_priors_jacobian = self._calc_parameters_prior_jacobian()
            project_jacobian = np.vstack((project_jacobian, parameter_priors_jacobian))

        if self.use_scale_factors_priors and len(self.scale_factors):
            sf_priors_jacobian = self._calc_scale_factors_prior_jacobian()
            project_jacobian = np.vstack((project_jacobian, sf_priors_jacobian))

        return project_jacobian

    def calc_rss_gradient(self, project_param_vector, *args):
        """
        Returns the gradient of the cost function (the residual sum of squares).

        The cost function is:

        .. math::
            C(\\theta)= 0.5*(\\sum{BY(\\theta)_{sim} - Y_{exp}})^2

        This function returns

        .. math::
            \\frac{\\partial C}{\\partial \\theta}

        The *args parameter is there for compatibility with the ampgo solver.

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
        residuals = self(project_param_vector)
        jacobian_matrix = self.calc_project_jacobian(project_param_vector)
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

    def calc_scale_factors_entropy(self, temperature=1.0):
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
        entropy: float
            The sum of the entropy of all scale factors in the model
        """
        entropy = 0.0
        for measure_name in self._measurement_to_model_map:
            if self.use_scale_factors[measure_name]:
                sf_iter = self.measure_iterator(measure_name)
                entropy += temperature * self._scale_factors[measure_name].calc_scale_factor_entropy(sf_iter,
                                                                                                     temperature)
        return entropy

    def free_energy(self, project_param_vector, temperature=1):
        """
        Calculates the free energy (see Sethna papers) of a particular parameter set.

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        free_energy : float
            The free energy of the project

        """
        rss = self.calc_sum_square_residuals(project_param_vector)
        entropy = self.calc_scale_factors_entropy(temperature)
        free_energy = rss - temperature * entropy
        return free_energy

    ##########################################################################################################
    # Saving and Loading Simulations, and fancy outputs
    ##########################################################################################################

    def project_param_dict_to_vect(self, param_dict, default_value=0.0):
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

    def group_experiments(self, settings_groups):
        """
        Returns a dictionary with experiments with same settings grouped together
        """
        grouped_experiments = defaultdict(list)

        for exp_idx, experiment in enumerate(self._experiments):
            group = []
            for setting in settings_groups:
                if setting not in experiment.settings:
                    raise KeyError('%s is not a setting in experiment %s' % (setting, experiment.name))
                else:
                    group.append(experiment.settings[setting])
            group = tuple(group)  # So we can hash it
            grouped_experiments[group].append(experiment)

        return grouped_experiments

    def plot_experiments(self, settings_groups=None, labels=None, use_experimental_timepoints=True):
        # TODO: fix labels
        import matplotlib.pyplot as plt
        from matplotlib.colors import rgb2hex
        import seaborn as sns

        if not use_experimental_timepoints and self._project_param_vector is not None:
            self._sim_experiments(use_experimental_timepoints=False)
            # We simulate again, this time with all timepoints, after SF are already calculated.

        # We try to group together experiments according to the settings.
        if settings_groups is not None:
            grouped_experiments = self.group_experiments(settings_groups)
        else:
            grouped_experiments = {e.name: e for e in self.experiments}
            # Each experiment by itself

        for group in grouped_experiments:
            # First, count how many different measurements there
            measured_variables = set()

            for experiment in grouped_experiments[group]:
                for measurement in experiment.measurements:
                    measured_variables.add(measurement.variable_name)

            n_measures = len(measured_variables)
            fig, axs = plt.subplots(n_measures)
            if n_measures == 1:
                axs = [axs]

            for measure_name, ax in zip(measured_variables, axs):
                ax.hold(True)
                sf = self._scale_factors[measure_name].sf
                palette = sns.color_palette("hls", len(grouped_experiments[group]))
                # For each group, we want to plot different variables in different groups
                ymax = 0

                for c_idx, experiment in enumerate(grouped_experiments[group]):
                    color = rgb2hex(palette[c_idx])

                    measurement = experiment.get_variable_measurements(measure_name)
                    measurement.plot_measurement(ax=ax, color=color, marker='o', linestyle='--')
                    if np.max(measurement.values > ymax):
                        ymax = np.max(measurement.values)

                    exp_idx = self.get_experiment_index(experiment.name)

                    if self._project_param_vector is not None:
                        exp_sim = self._all_sims[exp_idx][measure_name]
                        sim_data = exp_sim['value']
                        sim_t = exp_sim['timepoints']
                        ax.plot(sim_t, sim_data * sf, color=color)

                        if np.max(sim_data * sf) > ymax:
                            ymax = np.max(sim_data * sf)

                ax.set_ylim((0, ymax))

            fig.suptitle(group.__repr__())

    def print_param_settings(self):
        """
        Prints out all the parameters combinations in the project in a fancy way
        """
        total_params = 0
        for p_group in self._project_param_idx:
            exp_settings = self._project_param_idx[p_group].keys()
            exp_settings = sorted(exp_settings)
            print '%s  total_settings: %d ' % (p_group, len(exp_settings))
            for exp_set in exp_settings:
                print '%s, measurements for param: %d \t' % (repr(exp_set), self._residuals_per_param[p_group][exp_set])
                total_params += 1
                print '\n***********************'

    def get_parameter_settings(self):
        """
        Returns the dictionary containing all the parameter settings, and their index in the parameter vector
        """
        return copy.deepcopy(self._project_param_idx)

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


    def print_project_report(self):
        """" Prints all project settings"""
        # TODO: Make a pandas DataFrame for nice formatting of parameter priors/settings/values, etc

        # First, experiments:
        print "Experiments In Project:"
        for e in self._experiments:
            print e.name

        print "________________________\n\n"
        print "Parameters in Project:"
        self.print_param_settings()

        print "________________________\n\n"
        print "Priors on Parameters:"
