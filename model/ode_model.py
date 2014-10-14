from collections import OrderedDict

import numpy as np
import numba

from scipy.integrate import odeint
from abstract_model import ModelABC
from numpy import log


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

    def __init__(self, model, sens_model, n_vars, param_order, model_name="Model",
                 use_jit=True):
        self._unjitted_model = model  # Keep unjitted version just in case
        self._unjitted_sens_model = sens_model
        if use_jit:
            model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(model)
            sens_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(sens_model)
            self._jit_enabled = True
        else:
            self._jit_enabled = False

        super(OdeModel, self).__init__(model, n_vars, model_name)
        self.sens_model = sens_model
        self.param_order = param_order

    def get_n_vars(self):
        return self._n_vars

    n_vars = property(get_n_vars)

    def enable_jit(self):
        if self._jit_enabled:
            print "Model is already JIT'ed using Numba"
        else:
            self._model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(self._model)
            self.sens_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(self.sens_model)
            self._jit_enabled = True

    def calc_jacobian(self, experiment_params, t_sim,
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

        experiment_params = OdeModel.param_transform(experiment_params)
        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.sens_model(y, t, yout, experiment_params)
            return yout

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim)
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)
        return jacobian_sim

    def simulate_experiment(self, experiment_params, t_sim, init_conditions=None):
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

        experiment_params = OdeModel.param_transform(experiment_params)
        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self._model(y, t, yout, experiment_params)
            return yout

        model_sim = odeint(func_wrapper, init_conditions, t_sim)

        return model_sim


    @staticmethod
    def param_transform(project_param_vector):
        """
        Sometimes, it's convenient to optimize models in log-space to avoid negative values.
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