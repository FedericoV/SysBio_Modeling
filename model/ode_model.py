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

        super(OdeModel, self).__init__(model, n_vars, param_order, model_name)
        self.sens_model = sens_model

    def enable_jit(self):
        if self._jit_enabled:
            print "Model is already JIT'ed using Numba"
        else:
            self._model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(self._model)
            self.sens_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(self.sens_model)
            self._jit_enabled = True

    def calc_jacobian(self, experiment_params, t_sim, init_conditions):
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

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self.sens_model(y, t, yout, experiment_params)
            return yout

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim)
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)
        sensitivity_eqns = jacobian_sim[:, self.n_vars:]

        return sensitivity_eqns

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

        yout = np.zeros_like(init_conditions)

        def func_wrapper(y, t):
            self._model(y, t, yout, experiment_params)
            return yout

        model_sim = odeint(func_wrapper, init_conditions, t_sim)

        return model_sim