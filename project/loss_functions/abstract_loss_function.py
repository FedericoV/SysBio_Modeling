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


class DifferentiableLossFunctionABC(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableLossFunctionABC, self).__init__()

    def gradient(self, sim_values, exp_values, exp_std):
        pass

    def jacobian(self, sim_values, exp_values, exp_std):
        pass


