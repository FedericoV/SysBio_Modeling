from collections import OrderedDict

import numpy as np



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

    def __init__(self, model, sens_model, n_vars, param_order, model_name="Model",
                 use_jit='True', model_jac=None, sens_model_jac=None, jit_type='numba'):
        self._unjitted_model = model  # Keep unjitted version just in case
        self._unjitted_sens_model = sens_model
        self._unjitted_model_jac = model_jac
        self._unijitted_sens_model_jac = sens_model_jac
        self._jit_enabled = False

        super(OdeModel, self).__init__(model, n_vars, param_order, model_name)
        self.sens_model = sens_model
        self.model_jac = model_jac
        self.sens_model_jac = sens_model_jac
        self.use_jac = True
        self.jit_type = jit_type

        if use_jit:
            self.enable_jit()

    def enable_jit(self):
        if self.jit_type == 'numba':
            from numba import jit as numba_jit

            self._model = numba_jit("void(f8[:], f8, f8[:], f8[:])")(self._unjitted_model)
            self.sens_model = numba_jit("void(f8[:], f8, f8[:], f8[:])")(self._unjitted_sens_model)

            if self.model_jac is not None:
                self.model_jac = numba_jit("void(f8[:], f8, f8[:, :], f8[:])")(self._unjitted_model_jac)

            if self.sens_model_jac is not None:
                self.sens_model_jac = numba_jit("void(f8[:], f8, f8[:, :], f8[:])")(self._unijitted_sens_model_jac)

        elif self.jit_type == 'hope':
            from hope import jit as hope_jit

            self._model = hope_jit(self._unjitted_model)
            self.sens_model = hope_jit(self._unjitted_sens_model)

            if self.model_jac is not None:
                self.model_jac = hope_jit(self._unjitted_model_jac)

            if self.sens_model_jac is not None:
                self.sens_model_jac = hope_jit(self._unijitted_sens_model_jac)

            self._jit_enabled = True

    def calc_jacobian(self, experiment_params, t_sim, init_conditions):
        # TODO: BROKEN COMMENTS
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

        if self.sens_model_jac is None or not self.use_jac:
            jac_wrapper = None
        else:
            jacout = np.zeros((len(init_conditions), len(init_conditions)))
            def jac_wrapper(y, t):
                self.sens_model_jac(y, t, jacout, experiment_params)
                return jacout

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim, Dfun=jac_wrapper, col_deriv=True)
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

        if self.model_jac is None or not self.use_jac:
            jac_wrapper = None
        else:
            jacout = np.zeros((len(init_conditions), len(init_conditions)))
            def jac_wrapper(y, t):
                self.model_jac(y, t, jacout, experiment_params)
                return jacout

        yout = np.zeros_like(init_conditions)
        def func_wrapper(y, t):
            self._model(y, t, yout, experiment_params)
            return yout

        model_sim = odeint(func_wrapper, init_conditions, t_sim, Dfun=jac_wrapper, col_deriv=True)
        return model_sim