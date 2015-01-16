__author__ = 'Federico Vaggi'

import numpy as np

from ..abstract_loss_function import LossFunctionWithScaleFactors, DifferentiableLossFunctionABC
from .linear_scale_factor import LinearScaleFactor


class SquareLossFunction(LossFunctionWithScaleFactors, DifferentiableLossFunctionABC):
    def __init__(self, sf_groups=None, sf_type=LinearScaleFactor):
        """
        Default Square Loss Function:

        .. math::
            C(\\theta)= 0.5*(\\sum{BX_i - Y_i})^2

        Where:

        X_i is a v
        """
        super(SquareLossFunction, self).__init__(sf_groups, sf_type)

    def evaluate(self, simulations, experiment_measures):
        res_array = self.residuals(simulations, experiment_measures)
        return 0.5 * np.sum(res_array ** 2)

    def residuals(self, simulations, experiment_measures):
        if len(self._scale_factors) != 0:
            # Scale simulations by scale factor
            self.update_scale_factors(simulations, experiment_measures)
            self.update_sf_priors_residuals(simulations)  # We update simulations in place
            simulations = self.scale_sim_values(simulations)

        res = (simulations['mean'] - experiment_measures['mean']) / experiment_measures['std']

        return res

    def jacobian(self, simulations, experiment_measures, simulations_jacobian):
        if len(self._scale_factors) == 0:
            return simulations_jacobian

        self.update_scale_factors(simulations, experiment_measures)
        self.update_scale_factors_gradient(simulations, experiment_measures, simulations_jacobian)
        self.update_sf_priors_gradient(simulations_jacobian)

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

        return scaled_jacobian

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
