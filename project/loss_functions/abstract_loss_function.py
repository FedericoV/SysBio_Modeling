__author__ = 'Federico Vaggi'
from abc import ABCMeta, abstractmethod


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

    def __init__(self, scaled_measures_idx):
        super(LossFunctionWithScaleFactors, self).__init__()

        self.scale_factors_idx = scaled_measures_idx
        # Dict of indices of where we find the measurements of a given species

        self.scale_factors = {(measure_name, None) for measure_name in scaled_measures_idx}
        # Initialize the scale factors to none


class DifferentiableLossFunctionABC(LossFunctionABC):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(DifferentiableLossFunctionABC, self).__init__()

    def gradient(self, sim_values, exp_values, exp_std):
        pass

    def jacobian(self, sim_values, exp_values, exp_std):
        pass


