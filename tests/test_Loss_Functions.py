__author__ = 'Federico - Windows'

from unittest import TestCase

import numpy as np
import pandas as pd
from statsmodels.tools.numdiff import approx_fprime

from ..project.loss_functions.squared_loss import SquareLossFunction


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

    def test_unscaled_jacobian(self):
        def model_fcn(p):
            a, b, c = p
            y = a * np.sin(b * t) - c * (t ** 2.0)
            return y

        def model_jac(p):
            a, b, c = p
            jac = np.zeros((len(t), len(p)))
            jac[:, 0] = np.sin(b * t)  # da
            jac[:, 1] = a * t * np.cos(b * t)  # db
            jac[:, 2] = -t ** 2  # dc
            return jac

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

        def unscaled_residuals(new_p):
            new_vals = model_fcn(new_p)

            sim = pd.DataFrame(np.vstack((new_vals, t)).T, columns=['mean', 'timecourse'])
            sim.index = pd.MultiIndex.from_tuples([('Experiment_1', 'Val') for _ in range(11)])

            residuals = scaled_lf.residuals(sim, measures)
            return residuals

        scaled_num_jac = approx_fprime(p1, unscaled_residuals, centered=True)

        scaled_lf_jac = scaled_lf.jacobian(sim, measures, jac)

        assert np.allclose(scaled_lf_jac.values, scaled_num_jac, rtol=0.01)


