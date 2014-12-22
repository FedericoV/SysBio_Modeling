__author__ = 'Federico Vaggi'

from ..abstract_loss_function import LossFunctionWithScaleFactors, DifferentiableLossFunctionABC
from .linear_scale_factor import LinearScaleFactor
import numpy as np


class SquareLossFunction(LossFunctionWithScaleFactors, DifferentiableLossFunctionABC):

    def __init__(self, sf_groups=None):
        """
        Default Square Loss Function:

        .. math::
            C(\\theta)= 0.5*(\\sum{BX_i - Y_i})^2

        Where:

        X_i is a v
        """
        super(SquareLossFunction, self).__init__(sf_groups, LinearScaleFactor)

        for measure_group in sf_groups:
            self._scale_factors[measure_group] = LinearScaleFactor()

    def evaluate(self, simulations, experiment_measures):
        res_array = self.residuals(simulations, experiment_measures)
        return np.sum(res_array**2)

    def residuals(self, simulations, experiment_measures):
        self.update_scale_factors(simulations, experiment_measures)
        scaled_sim_values = self.scale_sim_values(simulations)
        # Scale residuals by scale factor
        return 1/2.0 * (scaled_sim_values['mean'] - experiment_measures['mean']) / experiment_measures['std']

    def jacobian(self, simulations, experiment_measures, simulations_jacobian):
        self.update_scale_factors(simulations, experiment_measures)
        self.update_scale_factors_gradient(simulations, experiment_measures, simulations_jacobian)

        if len(self._scale_factors) == 0:
            return simulations_jacobian

        else:
            scaled_jacobian = simulations_jacobian.copy()
            for measure_group in self._scale_factors:
                for measure in measure_group:
                    sf = self._scale_factors[measure].sf  # Scalar
                    sf_grad = self._scale_factors[measure].gradient  # Vector (n_params,)
                    measure_jac = simulations_jacobian.loc[(slice(None), measure), :].values[:, :-1]
                    # Matrix: (n_residuals, n_params)
                    measure_sim = simulations.loc[(slice(None), measure), :].values[:, 2]
                    # Vector: (n_residuals)

                    scaled_jac = measure_jac * sf + sf_grad * measure_sim
                    # J = dY_sim/dtheta * B + dB/dtheta * Y_sim

                    scaled_jacobian.loc[(slice(None), measure), :].values[:, :-1] = scaled_jac
            return scaled_jacobian
        # jac = measure_sim_jac * sf + sf_grad.T * measure_sim[:, np.newaxis]

    def scale_sim_values(self, simulations):
        if len(self._scale_factors) == 0:
            # Fast path
            return simulations

        else:
            scaled_sim_values = simulations.copy()
            for measure_group in self._scale_factors:
                for measure in measure_group:
                    sf = self._scale_factors[measure].sf
                    #  scaled_sim_values.loc[(slice(None), measure), 'mean'] *= sf  # TODO: Very slow.
                    scaled_sim_values.loc[(slice(None), measure), :].values[:, 2] *= sf  # TODO: Very slow.

        return scaled_sim_values

    def update_scale_factors(self, simulations, experiment_measures):
        if len(self._scale_factors) == 0:
            pass
        else:
            for measure_group in self._scale_factors:
                # Multiple measures can share the same scale factor
                group_sims = []
                group_exp_measures = []
                for measure in measure_group:
                    group_sims.append(simulations.loc[(slice(None), measure), :].values[:, 0])  # Values
                    group_exp_measures.append(experiment_measures.loc[(slice(None), measure), :].values[:, :2])
                    # Here we get values and std
                    # Probably slow indexing here.  Have to parse it carefully.

                group_sims = np.hstack(group_sims)
                group_exp_measures = np.hstack(group_exp_measures)
                self._scale_factors[measure_group].update_sf(group_sims, group_exp_measures[:, 0],
                                                             group_exp_measures[:, 1])

    def update_scale_factors_gradient(self, simulations, experiment_measures, simulations_jacobian):
        if len(self._scale_factors) == 0:
            pass
        else:
            for measure_group in self._scale_factors:
                # Multiple measures can share the same scale factor
                group_sims = []
                group_exp_measures = []
                for measure in measure_group:
                    group_sims.append(simulations.loc[(slice(None), measure), :].values[:, 0])  # Values
                    group_exp_measures.append(experiment_measures.loc[(slice(None), measure), :].values[:, :2])
                    # Here we get values and std
                    # Probably slow indexing here.  Have to parse it carefully.

                group_sims = np.hstack(group_sims)
                group_exp_measures = np.hstack(group_exp_measures)
                self._scale_factors[measure_group].update_sf_gradient(group_sims, exp_data=group_exp_measures[:, 0],
                                                                      group_exp_measures[:, 1])

