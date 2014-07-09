__author__ = 'Federico Vaggi'

from abc import ABCMeta, abstractmethod


class ScaleFactorABC(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, measure_names, log_prior=None, log_prior_sigma=None):
        self.measure_names = measure_names
        self._sf = None
        self._sf_gradient = None
        self.log_prior = log_prior
        self.log_prior_sigma = log_prior_sigma

    @abstractmethod
    def calc_sf(self, simulations, measurements):
        pass

    @abstractmethod
    def calc_sf_gradient(self, simulations, measurements):
        pass

    @abstractmethod
    def calc_sf_prior_residuals(self):
        return

    @abstractmethod
    def calc_sf_prior_gradient(self):
        return



