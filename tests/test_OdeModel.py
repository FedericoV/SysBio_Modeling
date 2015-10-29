from unittest import TestCase
from ..model import OdeModel
from test_utils.jittable_model import model as jitted_model
from test_utils.sens_jittable_model import sens_model as jitted_sens_model
import numpy as np


class TestOdeModel(TestCase):
    @classmethod
    def setUpClass(cls):
        # Model
        ordered_params = ['k_deg', 'k_synt']
        n_vars = 1
        cls.ode_model = OdeModel(jitted_model, jitted_sens_model, n_vars, ordered_params)

    def test_get_n_vars(self):
        ode_model = TestOdeModel.ode_model
        assert (ode_model.n_vars == 1)

    def test_simulate_experiment(self):
        param_vector = np.array([0.001, 0.01])
        timepoints = np.linspace(0, 100, 10)
        y_sim = TestOdeModel.ode_model.simulate(param_vector, timepoints)

        desired = np.array([0, 0.11049612, 0.21977129, 0.32783901, 0.43471262, 0.54040533,
                            0.64493016, 0.74830004, 0.85052773, 0.95162583])

        actual = y_sim[:, 0]
        assert np.allclose(actual, desired, rtol=0.05)

    def test_calc_jacobian(self):
        param_vector = np.array([0.001, 0.01])
        timepoints = np.linspace(0, 100, 10)
        init = np.zeros((3,))
        y_jac = TestOdeModel.ode_model.calc_jacobian(param_vector, timepoints, init)

        # Check with analytical derivative now
        k_deg = param_vector[0]
        k_synt = param_vector[1]

        # with respect to k_synt
        k_synt_analytical_jac = 1 / k_deg - np.exp(-k_deg * timepoints) / k_deg
        k_synt_numerical_jac = y_jac[:, 1]

        assert np.allclose(k_synt_analytical_jac, k_synt_numerical_jac, rtol=0.05)

        # with respect to k_deg
        k_deg_analytical_jac = k_synt * timepoints * np.exp(-k_deg * timepoints) / (
            k_deg - k_synt / k_deg ** 2 + k_synt * np.exp(-k_deg * timepoints) / k_deg ** 2)
        k_deg_numerical_jac = y_jac[:, 0]

        assert np.allclose(k_deg_analytical_jac, k_deg_numerical_jac, rtol=0.05)




