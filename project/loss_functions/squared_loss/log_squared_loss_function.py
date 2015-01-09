__author__ = 'Federico Vaggi'

import numpy as np

from .squared_loss import SquareLossFunction
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
        _j = super(LogSquareLossFunction, self).jacobian(simulations, experiment_measures, simulations_jacobian)

        return _j / simulations