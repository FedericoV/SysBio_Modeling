from unittest import TestCase
import os
from StringIO import StringIO
import imp

from nose.tools import raises
import numpy as np
from statsmodels.tools.numdiff import approx_fprime

from experiment.experiments import Experiment
from project.experiment_model_ensemble import SimpleProject
from symbolic.sympy_tools import make_sensitivity_model
from model import ode_model
from simple_model_settings import settings as experiment_settings

from jittable_model import model
from sens_jittable_model import sens_model


__author__ = 'Federico Vaggi'


def _simple_model_analytical_jac(k_deg, k_synt, t):
    k_synt_jac = k_synt * (1 / k_deg - np.exp(-k_deg * t) / k_deg)
    k_deg_jac = k_deg * (
    k_synt * t * np.exp(-k_deg * t) / k_deg - k_synt / k_deg ** 2 + k_synt * np.exp(-k_deg * t) / k_deg ** 2)
    return np.vstack((k_deg_jac, k_synt_jac))


class TestSimpleProject(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0., 11.11111111, 22.22222222, 33.33333333,
                                   44.44444444, 55.55555556, 66.66666667, 77.77777778,
                                   88.88888889, 100.])
        low_deg_measures = np.array([0., 0.11049608, 0.21977125, 0.32783897, 0.43471258, 0.54995561,
                                     0.65437492, 0.75764044, 0.85976492, 0.9516258]) * 3.75
        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': low_deg_measures}}
        exp_settings = {'Deg_Rate': 'Low'}
        low_deg = Experiment('Low_Deg_Exp', exp_data, experiment_settings=exp_settings)

        exp_timepoints = np.array([5.05050505, 9.09090909, 12.12121212, 16.16161616, 19.19191919,
                                   23.23232323, 26.26262626, 30.3030303, 33.33333333, 37.37373737,
                                   40.4040404, 44.44444444, 47.47474747, 50.50505051, 54.54545455,
                                   57.57575758, 61.61616162, 64.64646465, 68.68686869, 71.71717172,
                                   75.75757576, 78.78787879, 82.82828283, 85.85858586, 89.8989899,
                                   92.92929293])

        high_deg_measures = np.array([0.04925086, 0.08689927, 0.11415396, 0.14923229, 0.17462643, 0.20731014,
                                      0.23097075, 0.26142329, 0.28346868, 0.31184237, 0.33238284, 0.3588196,
                                      0.37795787, 0.39652489, 0.42042171, 0.43772125, 0.45998675, 0.47610533,
                                      0.49685087, 0.51186911, 0.53119845, 0.54519147, 0.56320128, 0.57623905,
                                      0.59301942, 0.60516718]) * 3.75
        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': high_deg_measures}}
        exp_settings = {'Deg_Rate': 'High'}
        high_deg = Experiment('High_Deg_Exp', exp_data.copy(), experiment_settings=exp_settings)
        experiments = [low_deg, high_deg]

        # Model
        ordered_params = ['k_deg', 'k_synt']
        n_vars = 1
        cls.ode_model = ode_model.OdeModel(model, sens_model, n_vars, ordered_params)

        measurement_variable_map = {'Variable_1': 0}
        proj = SimpleProject(cls.ode_model, experiments, experiment_settings, measurement_variable_map)
        cls.proj = proj

        global_param_vector = np.zeros((3,))
        low_deg_idx = proj.global_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj.global_param_idx['Group_1'][('High',)]
        synt_idx = proj.global_param_idx['k_synt']['Global']

        global_param_vector[high_deg_idx] = 0.01
        global_param_vector[low_deg_idx] = 0.001
        global_param_vector[synt_idx] = 0.01

        log_global_param_vector = np.log(global_param_vector)
        cls.log_global_param_vector = log_global_param_vector

    def test__project_initialization(self):
        proj = TestSimpleProject.proj
        assert (0 in proj.measurements_idx['Variable_1'])
        assert (1 in proj.measurements_idx['Variable_1'])
        global_param_idx = proj.global_param_idx
        assert (global_param_idx['k_synt'].keys() == ['Global'])
        assert (len(global_param_idx['Group_1'].keys()) == 2)  # Two settings for k_deg

        for exp in proj.experiments:
            setting = exp.settings['Deg_Rate']
            assert (global_param_idx['Group_1'][(setting,)] == exp.param_global_vector_idx['k_deg'])
            # Check parameters are set properly

        assert (proj.n_residuals == 35)

    def test_set_measurement_idx(self):
        proj = TestSimpleProject.proj
        assert (0 in proj.measurements_idx['Variable_1'])
        assert (1 in proj.measurements_idx['Variable_1'])

    @raises(ValueError)
    def test_null_param_vector(self):
        proj = TestSimpleProject.proj
        proj.global_param_vector = None
        proj._sim_experiments()

    def test_sim_experiments(self):
        proj = TestSimpleProject.proj
        log_global_param_vector = TestSimpleProject.log_global_param_vector
        proj(log_global_param_vector)
        assert (len(proj.all_sims) == 2)

        for exp_idx, exp in enumerate(proj.experiments):
            exp_data = exp.measurements['Variable_1']['value'] / 3.75
            exp_t = exp.measurements['Variable_1']['timepoints']
            exp_data = exp_data[exp_t != 0]
            sim = proj.all_sims[exp_idx]['Variable_1']
            assert (np.allclose(exp_data, sim['value'], rtol=0.05))

        assert (np.allclose(proj.scale_factors['Variable_1'], 3.75, rtol=0.05))
        # Excellent

        for exp_idx, res_block in enumerate(proj.all_residuals):
            exp = proj.experiments[exp_idx]
            measurement = exp.measurements['Variable_1']['value']
            total_measurement = np.sum(measurement)
            total_res = np.sum(res_block['Variable_1'])
            assert (total_measurement * 0.01 > total_res)

    def test_model_jacobian(self):
        proj = TestSimpleProject.proj
        log_global_param_vector = TestSimpleProject.log_global_param_vector
        global_param_vector = np.exp(log_global_param_vector)
        proj.global_jacobian(log_global_param_vector)

        for exp_idx, jac_block in enumerate(proj.model_jacobian):
            exp = proj.experiments[exp_idx]
            k_synt_idx = exp.param_global_vector_idx['k_synt']
            k_deg_idx = exp.param_global_vector_idx['k_deg']
            k_synt = global_param_vector[k_synt_idx]
            k_deg = global_param_vector[k_deg_idx]
            exp_t = exp.measurements['Variable_1']['timepoints']
            exp_t = exp_t[exp_t != 0]

            anal_jac = _simple_model_analytical_jac(k_deg, k_synt, exp_t)
            k_synt_analytical_jac = anal_jac[1, :]
            k_deg_analytical_jac = anal_jac[0, :]

            k_synt_sensitivity_jac = jac_block['Variable_1'][:, k_synt_idx]
            k_deg_sensitivity_jac = jac_block['Variable_1'][:, k_deg_idx]

            assert np.allclose(k_synt_analytical_jac, k_synt_sensitivity_jac, rtol=0.05)
            assert np.allclose(k_deg_analytical_jac, k_deg_sensitivity_jac, rtol=0.05)

    def test_scale_factor_jacobian(self):
        proj = TestSimpleProject.proj
        log_global_param_vector = TestSimpleProject.log_global_param_vector
        proj.global_jacobian(log_global_param_vector)
        scale_factors_jacobian = proj.scale_factors_jacobian['Variable_1']

        def get_scale_factors(x):
            proj(x)
            return proj.scale_factors['Variable_1']

        num_scale_factors = approx_fprime(log_global_param_vector, get_scale_factors, centered=True)
        assert np.allclose(scale_factors_jacobian, num_scale_factors, rtol=0.01)

    def test_global_jacobian(self):
        # Known test failure.  Most likely due to numerical failures in scaling factor.
        proj = TestSimpleProject.proj
        global_param_vector = np.zeros((3,))
        low_deg_idx = proj.global_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj.global_param_idx['Group_1'][('High',)]
        synt_idx = proj.global_param_idx['k_synt']['Global']

        global_param_vector[high_deg_idx] = 0.01
        global_param_vector[low_deg_idx] = 0.001
        global_param_vector[synt_idx] = 0.01
        log_global_param_vector = np.log(global_param_vector)

        sens_jacobian = proj.global_jacobian(log_global_param_vector)

        def get_scaled_sims(x):
            proj(x)
            sims = []
            for sim in proj.all_sims:
                exp_sim = sim['Variable_1']['value']
                sims.extend(exp_sim.tolist())
            sims = np.array(sims)
            scale = proj.scale_factors['Variable_1']
            return sims * scale

        num_global_jac = approx_fprime(log_global_param_vector, get_scaled_sims, centered=True)
        assert (np.sum((num_global_jac - sens_jacobian) ** 2) < 1e-8)

        global_param_vector[high_deg_idx] = 0.02
        global_param_vector[low_deg_idx] = 0.003
        global_param_vector[synt_idx] = 0.05
        log_global_param_vector = np.log(global_param_vector)

        sens_rss_jac = proj.flat_jacobian(log_global_param_vector)
        num_rss_jac = approx_fprime(log_global_param_vector, proj.sum_square_residuals, centered=True)
        assert np.allclose(sens_rss_jac, num_rss_jac, atol=0.000001)


    @raises(AssertionError)
    def test_scale_factors_change(self):
        proj = TestSimpleProject.proj
        global_param_vector = np.zeros((3,))
        low_deg_idx = proj.global_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj.global_param_idx['Group_1'][('High',)]
        synt_idx = proj.global_param_idx['k_synt']['Global']

        global_param_vector[high_deg_idx] = 0.01
        global_param_vector[low_deg_idx] = 0.001
        global_param_vector[synt_idx] = 0.01

        log_global_param_vector = np.log(global_param_vector)
        mod_param_vector = np.copy(log_global_param_vector)
        mod_param_vector[0] += 0.2

        proj(log_global_param_vector)
        old_scale_factor = proj.scale_factors['Variable_1'].copy()
        # Changing param vector:
        proj(mod_param_vector)
        new_scale_factor = proj.scale_factors['Variable_1'].copy()

        assert np.allclose(old_scale_factor, new_scale_factor, rtol=0.0001)

    def test_optimization(self):
        from leastsq_mod import leastsq as geo_leastsq
        # Known test failure.  Most likely due to numerical failures in scaling factor.
        proj = TestSimpleProject.proj

        base_guess = np.log(np.ones((3,))*0.01)

        out = geo_leastsq(proj, base_guess, Dfun=proj.global_jacobian)
        #print '\n\n'
        #print np.exp(out)
        #print proj.scale_factors['Variable_1']

        #print np.sum(proj(base_guess))
        #print np.sum(proj(out))

    def test_experiment_settings(self):
        pass
