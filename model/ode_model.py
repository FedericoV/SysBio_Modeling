from collections import OrderedDict

import numpy as np
import numba
from scipy.integrate import odeint

from abstract_model import ModelABC


class OdeModel(ModelABC):

    def __init__(self, model, sens_model, n_vars, param_order,
                 use_jit=True):
        """
        OdeModel is the default and simplest class that it is in charge of simulating models of differential
        equations.  OdeModel is also able of using sensitivity equations (currently only forward sensitivity) to
        calculate the jacobian explicitly.

        :param model: callable function that computes the time derivative at t with parameters p.  Due to optional
            usage with numba, func signature is slightly different from that used by default by scipy.odeint
        :type model: callable(y, t, yout, p)
        :param sens_model: callable function that computes the time derivative at t with parameters p as well
            as all the sensitivity equations.  Can be generated from :model using functions from symbolic tool module.
            Note, y and yout are of size n_vars + (n_vars * n_params)
        :type sens_model: callable(y, t, yout, p)
        :param int n_vars: the number of state variables in the model
        :param param_order: the order in which the parameters appear in the model
        :type param_order: list or iterable with stable iterable order
        :param bool use_jit: whether to use numba to compile the callable functions to byte code
        """
        if use_jit:
            model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(model)
            sens_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(sens_model)
        super(OdeModel, self).__init__(model, n_vars)
        self.sens_model = sens_model
        self.param_order = param_order

    def get_n_vars(self):
        return self._n_vars

    n_vars = property(get_n_vars)

    @staticmethod
    def param_transform(global_param_vector):
        """
        :rtype:  np.array
        :param np.array global_param_vector: Input vector of parameters
        :return: Parameter vector after undergoing an exponential transform.  Mostly easier to work in logspace
        """
        exp_param_vector = np.exp(global_param_vector)
        return exp_param_vector

    @staticmethod
    def param_transform_derivative(global_param_vector):
        """
        :rtype:  np.array
        :param np.array global_param_vector: Input vector of parameters
        :return: Parameter vector after undergoing an exponential transform.  Mostly easier to work in logspace
        """
        return np.exp(global_param_vector)

    def simulate_experiment(self, global_param_vector, experiment, variable_idx,
                            all_timepoints=False):
        """
        Returns a list containing the experiment simulated at the timepoints of the measurements.

        :rtype : np.array
        :param np.array global_param_vector:
        :param experiment.Experiment experiment:
        :param dict variable_idx:
        """
        init_conditions = np.zeros((self._n_vars,))
        t_end = experiment.get_unique_timepoints()[-1]
        transformed_params = OdeModel.param_transform(global_param_vector)
        experiment_params = self.global_to_experiment_params(transformed_params, experiment)

        t_sim = np.linspace(0, t_end, 1000)
        # Note we have to begin the simulation at t-0 - but then we don't consider it.

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.model(y, t, yout, experiment_params)
            return yout

        y_sim = odeint(func_wrapper, init_conditions, t_sim)

        exp_sim = OrderedDict()
        for measure_name in experiment.measurements:
            exp_sim[measure_name] = {}
            v_idx = variable_idx[measure_name]
            measure_sim = y_sim[:, v_idx]

            if not all_timepoints:
                exp_timepoints = experiment.measurements[measure_name]['timepoints']
                exp_timepoints = exp_timepoints[exp_timepoints != 0]
                exp_t_idx = np.searchsorted(t_sim, exp_timepoints)
                exp_t = np.take(t_sim, exp_t_idx)
                measure_sim = np.take(measure_sim, exp_t_idx)
            else:
                exp_t = t_sim

            exp_sim[measure_name]['value'] = measure_sim
            exp_sim[measure_name]['timepoints'] = exp_t
        return exp_sim

    def _jacobian_sim_to_dict(self, global_param_vector, jacobian_sim, t_sim, experiment, variable_idx):
        """
        This maps the jacobian with respect to model parameters to a jacobian with respect to the global parameters
        being optimized.  Although a model might only have 20 parameters, the effective global parameters might be
        a lot more, because a model parameter is allowed to take on different values in different experiments, if it is
        local (unique to each experiment) or shared (fixed across multiple experiments, depending on the experiment
        settings).  Note that it is also possible for multiple parameters within a model to refer to the same global
        parameter, if those parameters are part of the same shared block and have the same dependencies.

        :param np.array global_param_vector:
        :param np.array jacobian_sim:
        :param np.array t_sim:
        :param Experiment experiment:
        :param dict variable_idx:
        :return: dictionary of jacobian with respect to each measurement in experiment
        """

        n_vars = self._n_vars
        y_sim_sens = jacobian_sim[:, n_vars:]
        glob_parameter_indexes = experiment.param_global_vector_idx
        n_exp_params = len(glob_parameter_indexes)

        jacobian_dict = OrderedDict()
        for measurement in experiment.measurements:
            v_idx = variable_idx[measurement]
            v_0 = v_idx * n_exp_params

            exp_timepoints = experiment.measurements[measurement]['timepoints']
            exp_timepoints = exp_timepoints[exp_timepoints != 0]
            exp_t_idx = np.searchsorted(t_sim, exp_timepoints)

            local_sens = y_sim_sens[exp_t_idx, v_0:(v_0+n_exp_params)]
            var_jacobian = np.zeros((len(exp_t_idx), len(global_param_vector)))

            for l_idx, p_name in enumerate(self.param_order):
                try:
                    g_idx = glob_parameter_indexes[p_name]
                except KeyError:
                    if p_name in experiment.fixed_parameters:
                        continue
                    else:
                        raise KeyError('%s not in %s fixed parameters.')
                # l_idx is the index of a parameter in the local Jacobian
                # p_name is the name of the parameter
                # g_idx is the index of a parameter in the global Jacobian
                var_jacobian[:, g_idx] += local_sens[:, l_idx]
            jacobian_dict[measurement] = var_jacobian * OdeModel.param_transform_derivative(global_param_vector)

        return jacobian_dict

    def global_to_experiment_params(self, global_param_vector, experiment):
        """
        :rtype np.array
        :param np.array global_param_vector: The vector of all parameters that are being optimized across all
            experiments
        :param Experiment experiment: The experiment for which we would like to extra the specific parameters
        :return: A vector of parameters specific to simulating that experiment
        """
        exp_param_vector = np.zeros((len(self.param_order),))
        for p_model_idx, p_name in enumerate(self.param_order):
            try:
                global_idx = experiment.param_global_vector_idx[p_name]
                param_value = global_param_vector[global_idx]
            except KeyError:
                param_value = experiment.fixed_parameters[p_name]
                # If it's not an optimized parameter, it must be fixed by experiment
            exp_param_vector[p_model_idx] = param_value
        return exp_param_vector

    def calc_jacobian(self, global_param_vector, experiment, variable_idx):
        """
        :rtype dict
        :param np.array global_param_vector: The vector of all parameters that are being optimized across all
            experiments
        :param Experiment experiment: The experiment which we are trying to calculate the jacobian for
        :param dict variable_idx: The mapping from measurements in the experiment to variable index in the model
        :return: A dictionary containing the partial derivative of the measurements in the model (mapped to model
            variables through variable_idx) to all the non-fixed parameters in the model.
        """

        transformed_params = OdeModel.param_transform(global_param_vector)
        experiment_params = self.global_to_experiment_params(transformed_params, experiment)
        t_end = experiment.get_unique_timepoints()[-1]
        t_sim = np.linspace(0, t_end, 1000)

        glob_parameter_indexes = experiment.param_global_vector_idx
        n_exp_params = len(glob_parameter_indexes)
        n_vars = self._n_vars

        init_conditions = np.zeros((n_vars + n_exp_params * n_vars,))
        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.sens_model(y, t, yout, experiment_params)
            return yout

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim)
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)
        jacobian_dict = self._jacobian_sim_to_dict(global_param_vector, jacobian_sim, t_sim, experiment, variable_idx)
        return jacobian_dict
