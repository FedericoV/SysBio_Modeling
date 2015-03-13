__author__ = 'Federico Vaggi'
from abc import ABCMeta
from copy import deepcopy
import numpy as np

from ..utils import OrderedHashDict


class LossFunctionABC(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def evaluate(self, simulations, experiment_measures):
        pass

    def residuals(self, simulations, experiment_measures):
        pass


class LossFunctionWithScaleFactors(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self, sf_groups, SF):
        # SF is a scale factor constructor
        super(LossFunctionWithScaleFactors, self).__init__()
        self._scale_factors = OrderedHashDict()
        self._all_sf_measures = set()

        if sf_groups is not None:
            for measure_group in sf_groups:
                self._scale_factors[measure_group] = SF()

    @property
    def scale_factors(self):
        return deepcopy(self._scale_factors)

    def set_scale_factor_priors(self, measure_name, log_scale_factor_prior, log_sigma_scale_factor):
        try:
            self._scale_factors[measure_name].log_prior = log_scale_factor_prior
            self._scale_factors[measure_name].log_prior_sigma = log_sigma_scale_factor
        except KeyError:
            raise KeyError("%s not present as a scale factor" % measure_name)

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

    @staticmethod
    def _group_simulations_and_experiments(simulations, experiment_measures, measure_group):
        if type(measure_group) is str:
            _measure_group = [measure_group]
        else:
            _measure_group = measure_group
            # Hack to work around scale factor groups
            # TODO: Refactor OrderedHashDict

        # Multiple measures can share the same scale factor
        sf_group_sims = []
        sf_group_exp = []
        for measure in _measure_group:
            sf_group_sims.append(simulations.loc[(slice(None), measure), :].values[:, 0])  # Values
            sf_group_exp.append(experiment_measures.loc[(slice(None), measure), :].values[:, :2])
            # Here we get values and std
            # Probably slow indexing here.  Have to parse it carefully.

        sf_group_sims = np.hstack(sf_group_sims)
        sf_group_exp = np.vstack(sf_group_exp)
        sf_group_val = sf_group_exp[:, 0]
        sf_group_std = sf_group_exp[:, 1]

        assert len(sf_group_sims) == len(sf_group_exp)
        return sf_group_sims, sf_group_val, sf_group_std

    def update_scale_factors(self, simulations, experiment_measures):
        # Note - relies on carefully sorted dataframes!!
        for measure_group in self._scale_factors:
            sims, vals, stds = self._group_simulations_and_experiments(simulations, experiment_measures, measure_group)
            self._scale_factors[measure_group].update_sf(sims, vals, stds)

    def update_sf_priors_residuals(self, simulations):
        """Modifies simulations in-place"""
        for measure, sf in self._scale_factors.items():
            sf_res = sf.calc_sf_prior_residual()
            if sf_res is not None:
                try:
                    if type(measure) is str:
                        measure_name = measure
                    else:
                        measure_name = next(iter(measure))  # Have to fix later probably
                    simulations.loc[("~~SF_Prior", "~%s" % measure_name)].values[:, 0] = np.log(sf.sf)
                except KeyError:
                    raise KeyError("No prior in simulations for %s" % measure_name)

    def update_sf_priors_gradient(self, simulations_jacobian):
        """Modifies jacobian in-place"""

        for measure, sf in self._scale_factors.items():
            grad = sf.calc_sf_prior_gradient()
            if grad is not None:
                try:
                    if type(measure) is str:
                        measure_name = measure
                    else:
                        measure_name = next(iter(measure))  # Have to fix later probably
                    simulations_jacobian.ix[("~~SF_Prior", "~%s" % measure_name), :] = grad
                except KeyError:
                    raise KeyError("No prior in jacobian for %s" % measure_name)

    def total_sf_entropy(self, simulations, experiment_measures, t=1.0):
        """Calculates the total entropy from scale factors.
        Basically, a regularization term useful in making MCMC sampling 'well-behaved'
        """

        # Note - relies on carefully sorted dataframes!!
        entropy = 0.0

        for measure_group in self._scale_factors:
            sims, vals, stds = self._group_simulations_and_experiments(simulations, experiment_measures, measure_group)
            entropy += t*self._scale_factors[measure_group].calc_scale_factor_entropy(sims, vals, stds, t)

        return entropy


class DifferentiableLossFunctionABC(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableLossFunctionABC, self).__init__()

    def gradient(self, sim_values, exp_values, exp_std):
        pass

    def jacobian(self, sim_values, exp_values, exp_std):
        pass
