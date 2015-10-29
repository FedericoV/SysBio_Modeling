from collections import OrderedDict
from scipy.integrate import odeint
from abstract_model import ModelABC
import numpy as np


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
                 model_name='Model', use_jit=True, model_jac=None,
                 sens_model_jac=None, jit_type='numba'):
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

        else:
            raise ValueError('jit_type has to be "hope" or "numba"')

        self._jit_enabled = True

    def disable_jit(self):
        self._model = self._unjitted_model
        self.sens_model = self._unjitted_sens_model
        self.model_jac = self._unjitted_model_jac
        self.sens_model_jac = self._unijitted_sens_model_jac
        self._jit_enabled = False

    def calc_jacobian(self, experiment_params, t_sim, init_conditions):
        # TODO: BROKEN COMMENTS
        """
        Simulates the model including the sensitivity equations.

        Parameters
        ----------
        experiment_params: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters of the model
        t_sim: :class:`~numpy:numpy.ndarray`
            An (t,) dimensional array containing the timesteps at which to simulate the model
        init_conditions: :class:`~numpy:numpy.ndarray`
            An (k,) dimensional array containing the initial conditions of the model, where k = m + n*m
            where m is the number of state variables in the original model, and n is the number of parameters

        Returns
        -------
        sensitivity_eqns: :class:`~numpy:numpy.ndarray`
            An (len(t_sim), n*m) dimensional array containing the sensitivity of each of the m state variables
            with respect to the n-non fixed parameters

        Notes
        -----
        The jacobian is returned with respect to the model parameters, not the global parameters
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

        jacobian_sim = odeint(func_wrapper, init_conditions, t_sim, Dfun=jac_wrapper, col_deriv=True,
                              rtol=1e-10, atol=1e-10)
        # y_sim has dimensions (t_sim, n_vars + n_exp_params*n_vars)
        sensitivity_eqns = jacobian_sim[:, self.n_vars:]
        return sensitivity_eqns

    def simulate(self, experiment_params, t_sim, init_conditions=None):
        """
        Simulates the model using the `experiment` specific parameters.

        Evaluates :math:`Y_{sim}` using the parameters from `experiment`.

        Parameters
        ----------
        experiment_params: :class:`~numpy:numpy.ndarray`
            An (n,) dimensional array containing the parameters of the model
        t_sim: :class:`~numpy:numpy.ndarray`
            An (t,) dimensional array containing the timesteps at which to simulate the model
        init_conditions: :class:`~numpy:numpy.ndarray` , optional
            An (m,) dimensional array containing the initial conditions of the model, where m is the number
            of state variables in the model.
            Default: init_conditions = np.zeros((m,))

        Returns
        -------
        model_sim: :class:`~numpy:numpy.ndarray`
            An (len(t_sim), m) dimensional array containing the simulation at the timesteps
            specified by t_sim
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

        model_sim = odeint(func_wrapper, init_conditions, t_sim, Dfun=jac_wrapper, col_deriv=True,
                           rtol=1e-10, atol=1e-10)
        return model_sim
