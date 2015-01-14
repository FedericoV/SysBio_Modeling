__author__ = 'Federico - Windows'

from unittest import TestCase

import numpy as np
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime

from ..project.loss_functions.squared_loss import SquareLossFunction
from ..project.loss_functions.squared_loss import LogSquareLossFunction


__author__ = 'Federico Vaggi'


class TestLossFunction(TestCase):
    @classmethod
    def setUpClass(cls):
        t = np.linspace(0, 100, 101)
        lin = np.hstack((2 * t, t))
        square = np.hstack((t ** 2, t))
        sim = np.vstack((lin, square)).T

        index = [('Experiment 1', 'Lin') for d in range(101)]
        index.extend([('Experiment 2', 'Square') for d in range(101)])

        sim = pd.DataFrame(sim, columns=['mean', 'timecourse'])
        sim.index = pd.MultiIndex.from_tuples(index)
        sim.sortlevel(inplace=True)

        cls.sim = sim

    def test_no_scale_factors_loss(self):
        lf = SquareLossFunction()

        # Testing Residuals
        measures = TestLossFunction.sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))

        random_noise = np.random.randn(len(measures['mean']))
        measures['mean'] -= random_noise
        residuals = lf.residuals(TestLossFunction.sim, measures)
        assert np.allclose(random_noise, residuals)

        # Testing RSS (residual sum)
        total_noise = np.sum(random_noise ** 2)
        rss = lf.evaluate(TestLossFunction.sim, measures)
        assert ((0.5 * total_noise - rss) < 0.00001)

        # Testing residuals with non-uniform std
        measures['std'] = np.abs(np.random.randn(len(measures['std'])))
        varying_std_residuals = lf.residuals(TestLossFunction.sim, measures)
        scaled_random_noise = (TestLossFunction.sim['mean'] - measures['mean']) / measures['std']

        assert np.allclose(scaled_random_noise, varying_std_residuals)

    def test_scale_factors_loss(self):
        lf = SquareLossFunction(sf_groups=['Lin', 'Square'])

        measures = TestLossFunction.sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))

        measures.loc[(slice(None), 'Lin'), 'mean'] *= 2.0
        measures.loc[(slice(None), 'Square'), 'mean'] *= 3.6

        lf.update_scale_factors(TestLossFunction.sim, measures)

        assert (lf.scale_factors['Lin'].sf == 2.0)
        assert (lf.scale_factors['Square'].sf == 3.6)

        scaled_sim = lf.scale_sim_values(TestLossFunction.sim)

        assert np.allclose(measures.loc[(slice(None), 'Lin'), 'mean'],
                           scaled_sim.loc[(slice(None), 'Lin'), 'mean'])
        assert np.allclose(measures.loc[(slice(None), 'Square'), 'mean'],
                           scaled_sim.loc[(slice(None), 'Square'), 'mean'])

        residuals = lf.residuals(TestLossFunction.sim, measures)
        assert np.allclose(np.zeros_like(residuals), residuals)

    def test_sf_groups_set_correctly(self):
        lf = SquareLossFunction(['Variable_1'])
        sf_groups = lf.scale_factors

        assert ("Variable_1" in sf_groups)

        lf = SquareLossFunction(frozenset(['Variable_1', 'Variable_2']))
        sf_groups = lf.scale_factors

        assert ('Variable_1' in sf_groups)
        assert ('Variable_2' in sf_groups)

    def test_square_loss_jacobian(self):

        def model_fcn(p):
            a, b, c = p
            y = a * np.sin(b * t) - c * (t ** 2.0)
            return y

        def model_jac(p):
            a, b, c = p
            _jac = np.zeros((len(t), len(p)))
            _jac[:, 0] = np.sin(b * t)  # da
            _jac[:, 1] = a * t * np.cos(b * t)  # db
            _jac[:, 2] = -t ** 2  # dc
            return _jac

        p1 = (0.3, 0.5, 1.3)
        t = np.linspace(0, 10, 11)
        vals1 = model_fcn(p1)
        jac1 = model_jac(p1)
        data = np.vstack((vals1, t)).T

        sim = pd.DataFrame(data, columns=['mean', 'timecourse'])
        sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])

        jac = pd.DataFrame(jac1, columns=['a', 'b', 'c'], index=sim.index)

        # First - check that analytical jacobian is correct:
        num_jac = approx_fprime(p1, model_fcn, centered=True)
        assert np.allclose(jac.values, num_jac, rtol=0.01)

        measures = sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))

        lf = SquareLossFunction()
        lf_jac = lf.jacobian(sim, measures, jac)
        assert np.array_equal(jac.values, lf_jac.values)
        # Test that it remains unchanged.

        scaled_lf = SquareLossFunction(sf_groups=['Val'])
        measures['mean'] *= 5
        random_noise = np.random.randn(len(measures['mean']))
        measures['mean'] -= random_noise

        # Check Scale Factor Grad is correct
        def calc_sf(new_p):
            new_vals = model_fcn(new_p)

            _sim = pd.DataFrame(np.vstack((new_vals, t)).T, columns=['mean', 'timecourse'])
            _sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])
            scaled_lf.update_scale_factors(_sim, measures)
            return scaled_lf.scale_factors['Val'].sf

        num_sf_grad = approx_fprime(p1, calc_sf, centered=True)
        scaled_lf.update_scale_factors_gradient(sim, measures, jac)
        sf_grad = scaled_lf.scale_factors['Val'].gradient

        assert np.allclose(num_sf_grad, sf_grad, rtol=0.01)

        # Compare the finite differences jacobian with the model jacobian including SF
        def calc_residuals(new_p):
            new_vals = model_fcn(new_p)

            _sim = pd.DataFrame(np.vstack((new_vals, t)).T, columns=['mean', 'timecourse'])
            _sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])

            residuals = scaled_lf.residuals(_sim, measures)
            return residuals

        scaled_num_jac = approx_fprime(p1, calc_residuals, centered=True)
        scaled_lf_jac = scaled_lf.jacobian(sim, measures, jac)
        assert np.allclose(scaled_lf_jac.values, scaled_num_jac, rtol=0.01)

    def test_log_loss_residuals(self):
        log_lf = LogSquareLossFunction()

        # Testing Residuals
        sim = TestLossFunction.sim.copy()
        sim = sim[sim['mean'] != 0]
        # Removing zero values.  Those are a headache with logs.

        measures = sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))
        random_noise = 2.3
        measures['mean'] /= random_noise

        # Residuals be a constant value.
        sim_copy = sim.copy()
        measures_copy = measures.copy()
        residuals = log_lf.residuals(sim, measures)
        expected = np.ones_like(residuals) * np.log(random_noise)
        assert np.allclose(expected, residuals)

        # Check that running the loss function doesn't modify the simulations or measures
        assert (sim.all() == sim_copy.all()).all()
        assert (measures_copy.all() == measures.all()).all()

        # After scaling, the residuals should be near zero.
        log_lf = LogSquareLossFunction(sf_groups=['Lin', 'Square'])
        log_lf.update_scale_factors(sim, measures)
        log_sf_residuals = log_lf.residuals(sim, measures)
        assert np.allclose(log_sf_residuals, np.zeros_like(log_sf_residuals))

    def test_scale_factor_priors(self):
        lf = SquareLossFunction(sf_groups=['Lin', 'Square'])

        # Testing Residuals
        sim = TestLossFunction.sim.copy()
        sim = sim[sim['mean'] != 0]
        # Removing zero values.  Those are a headache with logs.

        measures = sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))
        measures['mean'] *= 5

        lf.set_scale_factor_priors('Square', 1.0, 2.0)

        assert (lf.scale_factors['Square'].log_prior == 1.0)
        assert (lf.scale_factors['Square'].log_prior_sigma == 2.0)

        res = lf.residuals(sim, measures)
        sf_prior_res = res[-1]

        assert np.allclose(sf_prior_res, ((np.log(5) - 1) / 2.0))

    def test_log_squared_loss_jacobian(self):
        # Test Jacobian:

        def log_model_fcn(p):
            a, b, c = p
            y = a + t*c*(b**2)
            return y

        def jac_log_model_fcn(p):
            a, b, c = p
            _jac = np.zeros((len(t), len(p)))
            _jac[:, 0] = 1
            _jac[:, 1] = 2*t*c*b
            _jac[:, 2] = t*(b**2)
            return _jac

        p1 = (0.3, 0.5, 1.3)
        t = np.linspace(0, 11, 11)
        vals1 = log_model_fcn(p1)
        jac1 = jac_log_model_fcn(p1)
        data = np.vstack((vals1, t)).T

        sim = pd.DataFrame(data, columns=['mean', 'timecourse'])
        sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])

        jac = pd.DataFrame(jac1, columns=['a', 'b', 'c'], index=sim.index)

        measures = sim.copy()
        measures.insert(1, 'std', np.ones_like(measures['mean']))

        scaled_log_lf = LogSquareLossFunction(sf_groups=['Val'])
        measures['mean'] *= 5
        random_noise = np.random.randn(len(measures['mean']))
        measures['mean'] += np.abs(random_noise)

        # Check Scale Factor Grad is correct
        def calc_log_sf(new_p):
            new_vals = log_model_fcn(new_p)

            _sim = pd.DataFrame(np.vstack((new_vals, t)).T, columns=['mean', 'timecourse'])
            _sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])
            scaled_log_lf.update_scale_factors(_sim, measures)
            return scaled_log_lf.scale_factors['Val'].sf

        num_log_sf_grad = approx_fprime(p1, calc_log_sf, centered=True)
        scaled_log_lf.update_scale_factors_gradient(sim, measures, jac)
        log_sf_grad = scaled_log_lf.scale_factors['Val'].gradient

        assert np.allclose(num_log_sf_grad, log_sf_grad, rtol=0.01)

        def calc_log_residuals(new_p):
            new_vals = log_model_fcn(new_p)

            _sim = pd.DataFrame(np.vstack((new_vals, t)).T, columns=['mean', 'timecourse'])
            _sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])
            residuals = scaled_log_lf.residuals(_sim, measures)
            return residuals

        scaled_num_jac = approx_fprime(p1, calc_log_residuals, centered=True)
        scaled_lf_jac = scaled_log_lf.jacobian(sim, measures, jac)

        assert np.allclose(scaled_lf_jac.values, scaled_num_jac, rtol=0.01)