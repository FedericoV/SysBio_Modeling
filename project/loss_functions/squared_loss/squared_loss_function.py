__author__ = 'Federico Vaggi'

from ..abstract_loss_function import LossFunctionWithScaleFactors, DifferentiableLossFunctionABC
import numpy as np


class SquareLossFunction(LossFunctionWithScaleFactors, DifferentiableLossFunctionABC):

    def __init__(self, scaled_measures_idx):
        super(SquareLossFunction, self).__init__()

    def __call__(self, sim_values, exp_values, exp_std):
        res_array = self.residuals(self, sim_values, exp_values, exp_std)
        return np.sum(res_array**2)

    def residuals(self, sim_values, exp_values, exp_std):
        scaled_sim_values = self.scale_sim_values(sim_values)
        # Scale residuals by scale factor
        return 1/2 * (scaled_sim_values - exp_values) / exp_std
