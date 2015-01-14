__author__ = 'Federico Vaggi'

import numpy as np

from .squared_loss_function import SquareLossFunction
from .log_scale_factor import LogScaleFactor


class LogSquareLossFunction(SquareLossFunction):
    def __init__(self, sf_groups=None):
        """
        Log Square Loss Function:

        .. math::
            C(\\theta)= 0.5*(\\sum{log(BX_i) - log(Y_i)})^2

        Where:

        X_i is a v
        """
        super(LogSquareLossFunction, self).__init__(sf_groups, LogScaleFactor)

    def residuals(self, simulations, experiment_measures):
        if experiment_measures.values[:, 0].min().min() <= 0:
            raise ValueError("LogSquare loss cannot handle measurements smaller or equal to zero")

        if len(self._scale_factors) != 0:
            # Scale simulations by scale factor
            self.update_scale_factors(simulations, experiment_measures)
            simulations = self.scale_sim_values(simulations)

        simulations = simulations.copy()
        experiment_measures = experiment_measures.copy()

        if "~Prior" in simulations.index.levels[0]:
            # We use drop to get a view of the dataframe without priors
            no_priors_sim = simulations.drop("~Prior", axis=0, level=0)
            no_priors_exp = experiment_measures.drop("~Prior", axis=0, level=0)
        else:
            # Drop returns a copy if no "~Prior" was present.
            no_priors_sim = simulations
            no_priors_exp = experiment_measures

        no_priors_sim.values[no_priors_sim.values[:, 0] == 0] += 1e-7
        # Work around zero values in simulations.

        no_priors_sim.values[:, 0] = np.log(no_priors_sim.values[:, 0])
        no_priors_exp.values[:, 0] = np.log(no_priors_exp.values[:, 0])

        res = (simulations['mean'] - experiment_measures['mean']) / experiment_measures['std']

        for measure, sf in self._scale_factors.items():
            sf_res = sf.calc_sf_prior_residual()
            if sf_res is not None:
                res.loc[("~Prior", "%s_SF" % measure)] = sf_res
        return res

    def jacobian(self, simulations, experiment_measures, simulations_jacobian):

        if len(self._scale_factors) == 0:
            return simulations_jacobian / simulations.values[:, 0]

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

                measure_scaled_jac = (measure_jac / measure_sim[:, np.newaxis]) + (sf_grad / sf)

                # J = (dY_sim/dtheta / Y_sim) + (dB/dtheta / B)
                scaled_jacobian.ix[(slice(None), measure), :] = measure_scaled_jac  # TODO: Very slow

        for measure, sf in self._scale_factors.items():
            sf_prior_grad = sf.calc_sf_prior_gradient()
            if sf_prior_grad is not None:
                scaled_jacobian.ix[("~Prior", "%s_SF" % measure), :] = sf_prior_grad

        return scaled_jacobian
