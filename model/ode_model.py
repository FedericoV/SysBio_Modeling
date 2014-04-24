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

    def _jacobian_sim_to_dict(self, global_param_vector, jacobian_sim, t_sim, experiment, variable_idx):
        """
        Map the jacobian with respect to model parameters to the jacobian with respect to the global parameters
        """

        n_vars = self._n_vars
        y_sim_sens = jacobian_sim[:, n_vars:]
        # There are n_var state variables + n_vars * n_exp_params sensitivity variables.
        n_exp_params = len(experiment.param_global_vector_idx)

        jacobian_dict = OrderedDict()
        for measurement in experiment.measurements:
            transformed_params_deriv = OdeModel.param_transform_derivative(global_param_vector)
            var_name = measurement.variable_name
            v_idx = variable_idx[var_name]
            # Mapping between experimental measurement and model variable
            v_0 = v_idx * n_exp_params

            _, _, exp_timepoints = measurement.get_nonzero_measurements()
            exp_timepoints = exp_timepoints[exp_timepoints != 0]
            exp_t_idx = np.searchsorted(t_sim, exp_timepoints)

            local_sens = y_sim_sens[exp_t_idx, v_0:(v_0+n_exp_params)]
            var_jacobian = np.zeros((len(exp_t_idx), len(global_param_vector)))

            for p_model_idx, p_name in enumerate(self.param_order):
                # p_model_idx is the index of a parameter in the model
                # p_name is the name of the parameter
                try:
                    global_idx = experiment.param_global_vector_idx[p_name]
                    # g_idx is the index of a parameter in the global vector
                except KeyError:
                    if p_name not in experiment.fixed_parameters:
                        raise KeyError('%s not in %s fixed parameters.')
                    else:
                        continue
                        # We don't calculate the jacobian wrt fixed parameters.
                var_jacobian[:, global_idx] += local_sens[:, p_model_idx]
            jacobian_dict[var_name] = var_jacobian * transformed_params_deriv

        return jacobian_dict

    def _global_to_experiment_params(self, global_param_vector, experiment):
        """
        Extracts the experiment-specific parameters from the global parameter vector.
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
        Calculates the jacobian of the model, evaluated using the `experiment` specific parameters.

        The jacobian of the model is: :math:`J = \\frac{\\partial Y_{sim}}{\\partial \\theta}`

        Since the model is often evaluated in log-space, the jacobian includes the chain rule term.

        :math:`Y_{sim}` is evaluated at all timepoints where there is a measurement.

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
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

        transformed_params = OdeModel.param_transform(global_param_vector)
        experiment_params = self._global_to_experiment_params(transformed_params, experiment)
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

    def simulate_experiment(self, global_param_vector, experiment, variable_idx,
                            all_timepoints=False):
        """
        Simulates the model using the `experiment` specific parameters.

        Evaluates :math:`Y_{sim}` using the parameters from `experiment`.

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project
        experiment: Experiment
            The experiment we wish to simulate
        variable_idx: dict
            A dict mapping the experimental measures to state variables of the model
        all_timepoints: bool, optional
            If false, the function is evaluated only for the timepoints for which there is
            an experimental measurement.

        Returns
        -------
        exp_sim: dict
            A dictionary containing the values of the simulation.
        """
        init_conditions = np.zeros((self._n_vars,))
        t_end = experiment.get_unique_timepoints()[-1]
        transformed_params = OdeModel.param_transform(global_param_vector)
        experiment_params = self._global_to_experiment_params(transformed_params, experiment)

        t_sim = np.linspace(0, t_end, 1000)
        # Note we have to begin the simulation at t-0 - but then we don't consider it.

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.model(y, t, yout, experiment_params)
            return yout

        y_sim = odeint(func_wrapper, init_conditions, t_sim)

        exp_sim = OrderedDict()
        for measurement in experiment.measurements:
            measure_name = measurement.variable_name
            exp_sim[measure_name] = {}
            v_idx = variable_idx[measure_name]
            measure_sim = y_sim[:, v_idx]

            if not all_timepoints:
                exp_timepoints = measurement.timepoints
                exp_timepoints = exp_timepoints[exp_timepoints != 0]
                exp_t_idx = np.searchsorted(t_sim, exp_timepoints)
                exp_t = np.take(t_sim, exp_t_idx)
                measure_sim = np.take(measure_sim, exp_t_idx)
            else:
                exp_t = t_sim

            exp_sim[measure_name]['value'] = measure_sim
            exp_sim[measure_name]['timepoints'] = exp_t
        return exp_sim

    @staticmethod
    def param_transform(global_param_vector):
        """
        Sometimes, it's convenient to optimize models in logspace to avoid negative values.
        Instead of doing :math:`Y_{sim}(\\theta)` we compute :math:`Y_{sim}(f(\\theta))`

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        transformated_parameters: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array the parameters after applying a transformation

        See Also
        --------
        param_transform_derivative

        """
        exp_param_vector = np.exp(global_param_vector)
        return exp_param_vector

    @staticmethod
    def param_transform_derivative(global_param_vector):
        """
        The derivative of the function applied to the parameters prior to the simulation.
        :math:`\\frac{\\partial f}{\\partial \\theta}`

        Parameters
        ----------
        global_param_vector: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters being optimized in the project

        Returns
        -------
        transformation_derivative: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array  containing the derivatives of the parameter transformation function

        See Also
        --------
        param_transform
        """
        transformation_derivative = np.exp(global_param_vector)
        return transformation_derivative