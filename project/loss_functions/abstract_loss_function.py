__author__ = 'Federico Vaggi'
from abc import ABCMeta, abstractmethod
from ..utils import OrderedHashDict


class LossFunctionABC(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    def __call__(self, sim_values, exp_values, exp_std):
        pass

    def residuals(self, sim_values, exp_values, exp_std):
        pass


class LossFunctionWithScaleFactors(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self, sf_groups, SF):
        # SF is a scale factor constructor
        super(LossFunctionWithScaleFactors, self).__init__()
        self._scale_factors = OrderedHashDict()

        for measure_group in sf_groups:
            self._scale_factors[measure_group] = SF()


class DifferentiableLossFunctionABC(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableLossFunctionABC, self).__init__()

    def gradient(self, sim_values, exp_values, exp_std):
        pass

    def jacobian(self, sim_values, exp_values, exp_std):
        pass


