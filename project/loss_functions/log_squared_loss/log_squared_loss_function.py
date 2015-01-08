__author__ = 'Federico Vaggi'

import numpy as np
import pandas as pd

from ..abstract_loss_function import LossFunctionWithScaleFactors, DifferentiableLossFunctionABC
from .linear_scale_factor import LinearScaleFactor


class LogSquareLossFunction(LossFunctionWithScaleFactors, DifferentiableLossFunctionABC):
    def __init__(self, sf_groups=None):
        """
        Log Square Loss Function:

        .. math::
            C(\\theta)= 0.5*(\\sum{log(BX_i) - log(Y_i)})^2

        Where:

        X_i is a v
        """
        super(LogSquareLossFunction, self).__init__(sf_groups, LinearScaleFactor)

    def evaluate(self, simulations, experiment_measures):
        res_array = self.residuals(simulations, experiment_measures)
        return 0.5 * np.sum(res_array ** 2)

    def residuals(self, simulations, experiment_measures):
        if len(self._scale_factors) != 0:
            # Scale simulations by scale factor
            self.update_scale_factors(simulations, experiment_measures)
            simulations = self.scale_sim_values(simulations)

        res = (np.log(simulations['mean']) - np.log(experiment_measures['mean'])) / experiment_measures['std']

        """
        all_sf_res = []
        sf_res_idx = []
        for measure, sf in self._scale_factors.items():
            sf_res = sf.calc_sf_prior_residual()
            if sf_res is not None:
                all_sf_res.append(sf_res)
                sf_res_idx.append(("Prior", measure))

        if len(all_sf_res) > 0:
            all_sf_res = pd.Series(all_sf_res, index=pd.MultiIndex.from_tuples(sf_res_idx))
            res = res.append(all_sf_res)
        """

        return res

    def jacobian(self, simulations, experiment_measures, simulations_jacobian):

        if len(self._scale_factors) == 0:
            # d/dx log(f(x)) = 1/x * df/dx
            return simulations_jacobian / simulations

        self.update_scale_factors(simulations, experiment_measures)
        self.update_scale_factors_gradient(simulations, experiment_measures, simulations_jacobian)

        scaled_jacobian = simulations_jacobian.copy()
        for measure_group in self._scale_factors:
            if type(measure_group) is str:
                _measure_group = [measure_group]
            else:
                _measure_group = measure_group
                # Hack to work around scale factor groups
                # TODO: Refactor OrderedHashDict

            for measure in _measure_group:
                sf = self._scale_factors[measure].sf  # Scalar
                sf_grad = self._scale_factors[measure].gradient  # Vector (n_params,)
                measure_jac = simulations_jacobian.loc[(slice(None), measure), :].values
                # Matrix: (n_residuals, n_params)
                measure_sim = simulations.loc[(slice(None), measure), :].values[:, 0]
                # Vector: (n_residuals)

                measure_scaled_jac = measure_jac * sf + measure_sim[:, np.newaxis] * sf_grad
                # J = dY_sim/dtheta * B + dB/dtheta * Y_sim
                scaled_jacobian.ix[(slice(None), measure), :] = measure_scaled_jac  # TODO: Very slow

        for measure, sf in self._scale_factors.items():
            sf_jac = sf.calc_sf_prior_gradient()
            if sf_jac is not None:
                scaled_jacobian.ix[("Prior", measure), :] = sf_jac

        return scaled_jacobian / simulations

    def scale_sim_values(self, simulations):
        scaled_sim_values = simulations.copy()
        for measure_group in self._scale_factors:
            if type(measure_group) is str:
                _measure_group = [measure_group]
            else:
                _measure_group = measure_group
                # Hack to work around scale factor groups
                # TODO: Refactor OrderedHashDict

            for measure in _measure_group:
                sf = self._scale_factors[measure].sf
                scaled_sim_values.loc[(slice(None), measure), 'mean'] *= sf  # TODO: Very slow.
        return scaled_sim_values

    def update_scale_factors(self, simulations, experiment_measures):
        # Note - relies on carefully sorted dataframes!!
        for measure_group in self._scale_factors:
            if type(measure_group) is str:
                _measure_group = [measure_group]
            else:
                _measure_group = measure_group
                # Hack to work around scale factor groups
                # TODO: Refactor OrderedHashDict

            # Multiple measures can share the same scale factor
            group_sims = []
            group_exp_measures = []
            for measure in _measure_group:
                group_sims.append(simulations.loc[(slice(None), measure), :].values[:, 0])  # Values
                group_exp_measures.append(experiment_measures.loc[(slice(None), measure), :].values[:, :2])
                # Here we get values and std
                # Probably slow indexing here.  Have to parse it carefully.

            group_sims = np.hstack(group_sims)
            group_exp_measures = np.vstack(group_exp_measures)

            assert len(group_sims) == len(group_exp_measures)

            self._scale_factors[measure_group].update_sf(group_sims, group_exp_measures[:, 0],
                                                         group_exp_measures[:, 1])

    def update_scale_factors_gradient(self, simulations, experiment_measures, simulations_jacobian):
        if len(self._scale_factors) == 0:
            pass
        else:
            for measure_group in self._scale_factors:
                if type(measure_group) is str:
                    _measure_group = [measure_group]
                else:
                    _measure_group = measure_group

                group_sims = []
                group_jacs = []
                group_exp_measures = []
                for measure in _measure_group:
                    group_sims.append(simulations.loc[(slice(None), measure), :].values[:, 0])  # Values
                    group_exp_measures.append(experiment_measures.loc[(slice(None), measure), :].values[:, :2])
                    # Here we get values and std

                    group_jacs.append(simulations_jacobian.loc[(slice(None), measure), :].values)

                group_sims = np.hstack(group_sims)
                group_jacs = np.hstack(group_jacs)
                group_exp_measures = np.hstack(group_exp_measures)
                self._scale_factors[measure_group].update_sf_gradient(group_sims, group_exp_measures[:, 0],
                                                                      group_exp_measures[:, 1], group_jacs)
