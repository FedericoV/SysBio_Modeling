__author__ = 'federico'

import os
import imp
import numpy as np

from experiment.experiments import Experiment
from unittest import TestCase
from symbolic.sympy_tools import make_sensitivity_model
from StringIO import StringIO
from model import ode_model
from collections import OrderedDict


class TestOdeModel(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': np.log(exp_measures)}}
        simple_exp = Experiment('Simple_Experiment', exp_data)
        cls.simple_exp = simple_exp
        cls.simple_exp.param_global_vector_idx = OrderedDict()
        cls.simple_exp.param_global_vector_idx['k_deg'] = 0
        cls.simple_exp.param_global_vector_idx['k_synt'] = 1
        cls.variable_idx = {'Variable_1': 0}

        model_file = os.path.join(os.getcwd(), 'tests', 'simple_model.py')
        model_fh = open(model_file)
        sens_model_string = StringIO()
        model_dict = make_sensitivity_model(model_fh, sens_model_string)
        ordered_params = model_dict['Parameters']
        exec sens_model_string.getvalue()
        # From the string, make the function, put it in scope

        model_module = imp.load_source('simple_model', model_file)
        simple_model = model_module.simple_model
        n_vars = model_module.n_vars
        cls.ode_model = ode_model.OdeModel(simple_model, make_bound_model, n_vars, ordered_params)

    def test_get_n_vars(self):
        ode_model = TestOdeModel.ode_model
        assert (ode_model.n_vars == 1)

    def test_inner_model_param_transform(self):
        out = TestOdeModel.ode_model.inner_model_param_transform(np.ones((10,)))
        assert np.alltrue(out == np.exp(np.ones((10,))))

    def test_inner_model_param_transform_derivative(self):
        out = TestOdeModel.ode_model.inner_model_param_transform_derivative(np.ones((10,)))
        assert np.alltrue(out == np.exp(np.ones((10,))))

    def test_simulate_experiment(self):
        variable_idx = TestOdeModel.variable_idx
        exp = TestOdeModel.simple_exp
        param_vector = np.log(np.array([0.001,  0.01]))
        y_sim = TestOdeModel.ode_model.simulate_experiment(param_vector, exp, variable_idx)
        assert ('Variable_1' in y_sim)

        desired = np.array([0.11049612,  0.21977129,  0.32783901,  0.43471262, 0.54040533,
                            0.64493016,  0.74830004,  0.85052773,  0.95162583])

        actual = y_sim['Variable_1']['value']
        assert np.allclose(actual, desired, rtol=0.05)

    def test_calc_jacobian(self):
        variable_idx = TestOdeModel.variable_idx
        exp = TestOdeModel.simple_exp
        param_vector = np.log(np.array([0.001,  0.01]))
        var_name = variable_idx.keys()[0]
        exp_t = exp.measurements[var_name]['timepoints']
        exp_t = exp_t[exp_t != 0]
        n_res = len(exp_t)

        y_sim = TestOdeModel.ode_model.calc_jacobian(param_vector, exp, variable_idx)
        assert (n_res == y_sim[var_name].shape[0])
        # The Jacobian should have dimensions (n_res, n_global_params)
        n_exp_par = len(exp.param_global_vector_idx)

        assert (n_exp_par == y_sim[var_name].shape[1])

        # Check with analytical derivative now
        exponential_param_vector = np.exp(param_vector)
        k_synt_idx = exp.param_global_vector_idx['k_synt']
        k_synt = exponential_param_vector[k_synt_idx]

        k_deg_idx = exp.param_global_vector_idx['k_deg']
        k_deg = exponential_param_vector[k_deg_idx]

        k_synt_analytical_jac = 1/k_deg - np.exp(-k_deg*exp_t)/k_deg
        k_synt_analytical_jac *= k_synt
        k_synt_numerical_jac = y_sim[var_name][:, k_synt_idx]

        assert np.allclose(k_synt_analytical_jac, k_synt_numerical_jac, rtol=0.05)

        k_deg_analytical_jac = k_synt*exp_t*np.exp(-k_deg*exp_t)/k_deg - k_synt/k_deg**2 + k_synt*np.exp(-k_deg*exp_t)/k_deg**2
        k_deg_analytical_jac *= k_deg
        k_deg_numerical_jac = y_sim[var_name][:, k_deg_idx]

        assert np.allclose(k_deg_analytical_jac, k_deg_numerical_jac, rtol=0.05)




