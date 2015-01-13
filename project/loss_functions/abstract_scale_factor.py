__author__ = 'Federico Vaggi'

from abc import ABCMeta, abstractmethod


class ScaleFactorABC(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, log_prior=None, log_prior_sigma=None):
        self._sf = None
        self._sf_gradient = None
        self.log_prior = log_prior
        self.log_prior_sigma = log_prior_sigma

    @abstractmethod
    def update_sf(self, simulations, measurements):
        pass

    @abstractmethod
    def update_sf_gradient(self, simulations, measurements):
        pass

    @abstractmethod
    def calc_sf_prior_residual(self):
        return

    @abstractmethod
    def calc_sf_prior_gradient(self):
        return

    def __repr__(self):
        output = "SF value: %.5e\n" % self._sf

        if self.log_prior is not None:
            output += "SF log prior: %.4f\n" % self.log_prior
            output += "SF log prior sigma: %.4f\n" % self.log_prior_sigma

        return output


