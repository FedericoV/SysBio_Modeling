import numpy as np
from scipy.integrate import odeint
from collections import OrderedDict
from abstract_model import ModelABC


class OdeModel(ModelABC):

    def __init__(self, model, sensitivity_model_factory, n_vars, param_order):
        super(OdeModel, self).__init__(model, n_vars)
        self.sensitivity_model_factory = sensitivity_model_factory
        self.param_order = param_order

    def get_n_vars(self):
        return self._n_vars

    n_vars = property(get_n_vars)

    @staticmethod
    def inner_model_param_transform(global_param_vector):
        exp_param_vector = np.exp(global_param_vector)
        return exp_param_vector

    @staticmethod
    def inner_model_param_transform_derivative(global_param_vector):
        """
        :param global_param_vector: np.array
        :rtype : np.array
        """
        return np.exp(global_param_vector)

    def simulate_experiment(self, global_param_vector, experiment, variable_idx,
                            all_timepoints=False):
        """
        Returns a list containing the experiment simulated
        at the timepoints of the measurements.

        :rtype : np.array
        :param global_param_vector: np.array
        :param experiment: Experiment()
        :param variable_idx: dict()
        :type experiment: object
        """
        init_conditions = np.zeros((self._n_vars,))
        t_end = experiment.get_unique_timepoints()[-1]
        global_param_vector = OdeModel.inner_model_param_transform(global_param_vector)

        t_sim = np.linspace(0, t_end, 1000)
        # Note we have to begin the simulation at t-0 - but then we don't consider it.
        y_sim = odeint(self.model, init_conditions, t_sim,
                       args=(experiment.param_global_vector_idx,
                             experiment.fixed_parameters, global_param_vector))

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

    def calc_jacobian(self, global_param_vector, experiment, variable_idx):
        """
        Returns the jacobian of the model, evaluated at global_param_vector,
        using the setting in experiment, for all the model variables in variable_idx.

        Jacobian is of size:
        """

        trans_params = OdeModel.inner_model_param_transform(global_param_vector)
        closed_model = self.sensitivity_model_factory(experiment.param_global_vector_idx,
                                                      experiment.fixed_parameters, trans_params)
        t_end = experiment.get_unique_timepoints()[-1]
        t_sim = np.linspace(0, t_end, 1000)

        glob_parameter_indexes = experiment.param_global_vector_idx
        n_exp_params = len(glob_parameter_indexes)
        n_vars = self._n_vars

        init_conditions = np.zeros((n_vars + n_exp_params * n_vars,))

        y_sim_sens = odeint(closed_model, init_conditions, t_sim)[:, n_vars:]
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)

        jacobian = OrderedDict()
        for measurement in experiment.measurements:
            v_idx = variable_idx[measurement]
            v_0 = v_idx * n_exp_params

            exp_timepoints = experiment.measurements[measurement]['timepoints']
            exp_timepoints = exp_timepoints[exp_timepoints != 0]
            exp_t_idx = np.searchsorted(t_sim, exp_timepoints)

            local_sens = y_sim_sens[exp_t_idx, v_0:(v_0+n_exp_params)]
            var_jacobian = np.zeros((len(exp_t_idx), len(global_param_vector)))

            for l_idx, p_name in enumerate(self.param_order):
                g_idx = glob_parameter_indexes[p_name]
                # l_idx is the index of a parameter in the local Jacobian
                # p_name is the name of the parameter
                # g_idx is the index of a parameter in the global Jacobian
                var_jacobian[:, g_idx] = local_sens[:, l_idx]
            jacobian[measurement] = var_jacobian * OdeModel.inner_model_param_transform_derivative(global_param_vector)

        return jacobian
