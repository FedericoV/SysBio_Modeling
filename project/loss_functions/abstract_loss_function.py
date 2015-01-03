__author__ = 'Federico Vaggi'
from abc import ABCMeta
from copy import deepcopy

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


class DifferentiableLossFunctionABC(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableLossFunctionABC, self).__init__()

    def gradient(self, sim_values, exp_values, exp_std):
        pass

    def jacobian(self, sim_values, exp_values, exp_std):
        pass


