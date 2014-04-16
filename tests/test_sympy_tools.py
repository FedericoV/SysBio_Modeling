__author__ = 'Federico Vaggi'


from symbolic import make_sensitivity_model
from unittest import TestCase
from nose.tools import raises
import simple_model
import os
import numpy as np
from scipy.integrate import odeint
import numba
from jittable_model import model as unjitted_model
from sens_jittable_model import sens_model as sens_unjitted_model


def make_func_wrapper(model, n_vars, p):
    yout = np.zeros((n_vars,))

    def func_wrapper(y, t):
        model(y, t, yout, p)
        return yout

    return func_wrapper


class TestSymPyTools(TestCase):
    @classmethod
    def setUpClass(cls):
        sens_jit_model = os.path.join(os.getcwd(), 'tests', 'sens_jittable_model.py')
        sens_jit_fh = open(sens_jit_model, 'w')
        make_sensitivity_model(simple_model, sens_jit_fh)

        jit_model = os.path.join(os.getcwd(), 'tests', 'jittable_model.py')
        jit_fh = open(jit_model, 'w')
        make_sensitivity_model(simple_model, jit_fh, calculate_sensitivities=False)

        p = np.array([0.01, 0.001])
        n_vars = 1
        init_conditions_no_sens = np.zeros((n_vars,))
        t_sim = np.linspace(0, 100, 100)
        y_sim_simple = odeint(simple_model.simple_model, t_sim, init_conditions_no_sens, args=(p,))

        cls.t_sim = t_sim
        cls.p = p
        cls.n_vars = 1
        cls.y_sim = y_sim_simple

    @raises(ValueError)
    def test_requires_sens_dir(self):
        make_sensitivity_model(simple_model)
        # No sensitivity Directory

    def test_simulate_unjitted_models(self):
        y_sim = TestSymPyTools.y_sim
        t_sim = TestSymPyTools.t_sim
        n_vars = TestSymPyTools.n_vars
        p = TestSymPyTools.p

        init_conditions_no_sens = np.zeros((n_vars,))
        nojit_wrapped = make_func_wrapper(unjitted_model, 1, p)
        y_sim_unjitted = odeint(nojit_wrapped, t_sim, init_conditions_no_sens)
        assert(np.allclose(y_sim, y_sim_unjitted, rtol=0.001))

        init_conditions_sens = np.zeros((n_vars + n_vars * len(p),))
        nojit_sens_wrapped = make_func_wrapper(sens_unjitted_model, 1, p)
        y_sim_unjitted_sens = odeint(nojit_sens_wrapped, t_sim, init_conditions_sens)[0, :]
        assert(np.allclose(y_sim, y_sim_unjitted_sens, rtol=0.001))

    def test_simulate_jitted_models(self):
        y_sim = TestSymPyTools.y_sim
        t_sim = TestSymPyTools.t_sim
        n_vars = TestSymPyTools.n_vars
        p = TestSymPyTools.p

        init_conditions_no_sens = np.zeros((n_vars,))
        jitted_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(unjitted_model)
        jit_wrapped = make_func_wrapper(jitted_model, 1, p)
        y_sim_jitted = odeint(jit_wrapped, t_sim, init_conditions_no_sens)
        assert(np.allclose(y_sim, y_sim_jitted, rtol=0.001))

        init_conditions_sens = np.zeros((n_vars + n_vars * len(p),))
        sens_jitted_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(sens_unjitted_model)
        jit_sens_wrapped = make_func_wrapper(sens_jitted_model, 1, p)
        y_sim_jitted_sens = odeint(jit_sens_wrapped, t_sim, init_conditions_sens)[0, :]
        assert(np.allclose(y_sim, y_sim_jitted_sens, rtol=0.001))
