__author__ = 'Federico Vaggi'

import numpy as np

from .squared_loss_function import SquareLossFunction
from .log_scale_factor import LogScaleFactor

import pandas as pd


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

        if len(self._scale_factors) != 0:
            # Scale simulations by scale factor
            self.update_scale_factors(simulations, experiment_measures)
            self.update_sf_priors_residuals(simulations)
            simulations = self.scale_sim_values(simulations)

        if simulations['mean'].isnull().any():
            # Check if there are NaN or zeros (which cause NaN with logs) in the simulations
            res = np.zeros_like(simulations['mean'])
            res.fill(np.inf)
            return res

        ################################################################################
        # In a Pandas DataFrame, there is no easy way to 'reverse' index - to say we want all elements except the
        # priors.  The easiest way to do this is to get a view by dropping the priors, then modify the view,
        # modifying the view modifies the original DataFrame.
        ################################################################################
        simulations = simulations.copy()
        experiment_measures = experiment_measures.copy()

        if ("~Prior" in simulations.index.levels[0]) or ("~~SF_Prior" in simulations.index.levels[0]):
            # We use drop to get a view of the dataframe without priors
            no_priors_sim = simulations.drop(["~Prior", "~~SF_Prior"], axis=0, level=0)
            no_priors_exp = experiment_measures.drop(["~Prior", "~~SF_Prior"], axis=0, level=0)
        else:
            # Drop returns a copy if no "~Prior" was present.
            no_priors_sim = simulations
            no_priors_exp = experiment_measures

        if no_priors_exp.values[:, 0].min().min() <= 0:
            raise ValueError("LogSquare loss cannot handle measurements smaller or equal to zero")
        # In measurements though, we cannot.

        no_priors_sim.values[:, 0] = np.log(no_priors_sim.values[:, 0])  # Logscale simulations

        no_priors_exp.values[:, 1] /= no_priors_exp.values[:, 0]  # Scale standard deviation by mean
        # if a = 10 +- 1, and b = log(a), then b = log(10) +- (1/10) (basic propagation of error)
        no_priors_exp.values[:, 0] = np.log(no_priors_exp.values[:, 0])  # Logscale mean

        return (simulations['mean'] - experiment_measures['mean']) / experiment_measures['std']

    def jacobian(self, simulations, experiment_measures, simulations_jacobian):

        if len(self._scale_factors) == 0:
            return simulations_jacobian / simulations.values[:, 0]

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

                measure_scaled_jac = (measure_jac / measure_sim[:, np.newaxis]) + (sf_grad / sf)

                # J = (dY_sim/dtheta / Y_sim) + (dB/dtheta / B)
                scaled_jacobian.ix[(slice(None), measure), :] = measure_scaled_jac  # TODO: Very slow

        # Now we add the scale factor priors in here.
        return scaled_jacobian
