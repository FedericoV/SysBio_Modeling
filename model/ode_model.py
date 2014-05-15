from collections import OrderedDict

import numpy as np
import numba
from scipy.integrate import odeint

from abstract_model import ModelABC


class OdeModel(ModelABC):
    """Differential-based Models integrated with SciPy LSODA wrapper

    Attributes
    ----------
    model : func (x0, t, xout, p)
        A callable function that computes the time derivative at timesteps t as a function of parameters p.\n
        Signature is slightly different from the `scipy.odeint` signature due to limitations of `numba`
    sens_model: func (x0, t, xout, p)
        A callable function that computes the sensitivity equations for model.  Can be generated using the tools
        in the `symbolic` package.
    n_vars : int
        The number of state variables in the model
    param_order : list
        Order in which the parameters appear in the model.  Important for returning the jacobian in the correct
        order.
    use_jit : bool , optional
        Whether or not to jit `sens_model` and `model` using numba
    """

    def __init__(self, model, sens_model, n_vars, param_order,
                 use_jit=True):
        if use_jit:
            model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(model)
            sens_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(sens_model)
        super(OdeModel, self).__init__(model, n_vars)
        self.sens_model = sens_model
        self.param_order = param_order

    def get_n_vars(self):
        return self._n_vars

    n_vars = property(get_n_vars)

    def _jacobian_sim_to_dict(self, project_param_vector, jacobian_sim, t_sim, experiment, mapping_struct):
        """
        Map the jacobian with respect to model parameters to the jacobian with respect to the global parameters
        """

        n_vars = self.n_vars
        model_jac = jacobian_sim[:, n_vars:]
        # There are n_var state variables + n_vars * n_exp_params sensitivity variables.
        transformed_params_deriv = OdeModel.param_transform_derivative(project_param_vector)

        jacobian_dict = OrderedDict()
        for measurement in experiment.measurements:
            measure_name = measurement.variable_name
            model_jac_to_measure_func = mapping_struct[measure_name]['model_jac_to_measure_jac_func']
            mapping_parameters = mapping_struct[measure_name]['parameters']
            measure_jac = model_jac_to_measure_func(model_jac, t_sim, experiment, measurement, mapping_parameters)

            var_jacobian = np.zeros((measure_jac.shape[0], len(project_param_vector)))

            for p_model_idx, p_name in enumerate(self.param_order):
                # p_model_idx is the index of a parameter in the model
                # p_name is the name of the parameter
                try:
                    p_project_idx = experiment.param_global_vector_idx[p_name]
                    # g_idx is the index of a parameter in the global vector
                except KeyError:
                    if p_name not in experiment.fixed_parameters:
                        raise KeyError('%s not in %s fixed parameters.')
                    else:
                        continue
                        # We don't calculate the jacobian wrt fixed parameters.
                var_jacobian[:, p_project_idx] += measure_jac[:, p_model_idx]
            jacobian_dict[measurement.variable_name] = var_jacobian * transformed_params_deriv

        return jacobian_dict

    def _global_to_experiment_params(self, project_param_vector, experiment):
        """
        Extracts the experiment-specific parameters from the global parameter vector.
        """

        exp_param_vector = np.zeros((len(self.param_order),))
        for p_model_idx, p_name in enumerate(self.param_order):
            try:
                global_idx = experiment.param_global_vector_idx[p_name]
                param_value = project_param_vector[global_idx]
            except KeyError:
                param_value = experiment.fixed_parameters[p_name]
                # If it's not an optimized parameter, it must be fixed by experiment
            exp_param_vector[p_model_idx] = param_value
        return exp_param_vector

    def calc_jacobian(self, project_param_vector, experiment, measurement_to_model_map, t_sim=None,
                      init_conditions=None):
        """
        Calculates the jacobian of the model, evaluated using the `experiment` specific parameters.

        The jacobian of the model is: :math:`J = \\frac{\\partial Y_{sim}}{\\partial \\theta}`

        Since the model is often evaluated in log-space, the jacobian includes the chain rule term.

        :math:`Y_{sim}` is evaluated at all timepoints where there is a measurement.

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project
        experiment: Experiment
            The specific experiment for which we want to create the parameter vector
        variable_idx: dict
            A dict mapping the experimental measures to state variables of the model

        Returns
        -------
        jacobian_dict: dict
            A dict with keys equal to the experimental measures, and values equal to the jacobian of the model
            with respect to the global parameters.

        Notes
        -----
        The jacobian is returned with respect to the global parameters, not the model parameters.\n
        The jacobian with respect to global parameters that aren't in the experiment will thus be zero.
        """

        transformed_params = OdeModel.param_transform(project_param_vector)
        experiment_params = self._global_to_experiment_params(transformed_params, experiment)

        glob_parameter_indexes = experiment.param_global_vector_idx
        n_exp_params = len(glob_parameter_indexes)
        n_vars = self._n_vars

        if init_conditions is None:
            init_conditions = np.zeros((n_vars + n_exp_params * n_vars,))
        if t_sim is None:
            t_end = experiment.get_unique_timepoints()[-1]
            t_sim = np.linspace(0, t_end, 1000)

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.sens_model(y, t, yout, experiment_params)
            return yout

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim)
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)
        jacobian_dict = self._jacobian_sim_to_dict(project_param_vector, jacobian_sim, t_sim, experiment,
                                                   measurement_to_model_map)
        return jacobian_dict

    def simulate_experiment(self, project_param_vector, experiment, mapping_struct, t_sim=None, init_conditions=None):
        """
        Simulates the model using the `experiment` specific parameters.

        Evaluates :math:`Y_{sim}` using the parameters from `experiment`.

        Parameters
        ----------
        project_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project
        experiment: Experiment
            The experiment we wish to simulate
        mapping_struct: dict
            A dict mapping the experimental measures to state variables of the model
        all_timepoints: bool, optional
            If false, the function is evaluated only for the timepoints for which there is
            an experimental measurement.

        Returns
        -------
        exp_sim: dict
            A dictionary containing the values of the simulation.
        """
        if init_conditions is None:
            init_conditions = np.zeros((self._n_vars,))
        if t_sim is None:
            t_end = experiment.get_unique_timepoints()[-1]
            t_sim = np.linspace(0, t_end, 1000)

        transformed_params = OdeModel.param_transform(project_param_vector)
        experiment_params = self._global_to_experiment_params(transformed_params, experiment)

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self._model(y, t, yout, experiment_params)
            return yout

        model_sim = odeint(func_wrapper, init_conditions, t_sim)

        mapped_sim = self._map_model_sim(model_sim, experiment, mapping_struct, t_sim)
        return mapped_sim

    def _map_model_sim(self, model_sim, experiment, mapping_struct, t_sim):
        exp_sim = OrderedDict()

        for measurement in experiment.measurements:
            measure_name = measurement.variable_name
            model_variables_to_measure_func = mapping_struct[measure_name]['model_variables_to_measure_func']
            mapping_parameters = mapping_struct[measure_name]['parameters']

            exp_sim[measure_name] = {}
            measure_sim, exp_t_idx = model_variables_to_measure_func(model_sim, t_sim, experiment, measurement,
                                                                     mapping_parameters)

            exp_sim[measure_name]['value'] = measure_sim
            exp_sim[measure_name]['timepoints'] = t_sim[exp_t_idx]
        return exp_sim

    @staticmethod
    def param_transform(project_param_vector):
        """
        Sometimes, it's convenient to optimize models in logspace to avoid negative values.
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

    @staticmethod
    def param_transform_derivative(project_param_vector):
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