__author__ = 'Federico Vaggi'

import numpy as np
import scipy

from ..abstract_scale_factor import ScaleFactorABC


class LogScaleFactor(ScaleFactorABC):

    def __init__(self, log_prior=None, log_prior_sigma=None):
        super(LogScaleFactor, self).__init__(log_prior, log_prior_sigma)
        self._sf = 0
        self._log_data_sum = None
        self._inv_std_sum = None

    def update_sf(self, sim_data, exp_data, exp_std):
        exp_std /= exp_data  # scaling of standard deviation by mean because of log
        if self._inv_std_sum is None:
            self._inv_std_sum = np.sum((1.0 / exp_std**2))
            self._log_data_sum = np.exp(np.sum((np.log(exp_data) / (exp_std**2))) / self._inv_std_sum)
            # We can split the summation to cache the terms that depend on measurements only

        log_sim_sum = np.exp(-np.sum((np.log(sim_data) / (exp_std**2))) / self._inv_std_sum)
        # Cached version - only have to recalculate this.

        self._sf = self._log_data_sum * log_sim_sum

    def update_sf_gradient(self, sim_data, exp_data, exp_std, sim_jac):
        """
        Analytically calculates the gradient of the scale factors for each measurement
        """
        exp_std /= exp_data  # scaling of standard deviation by mean because of log
        sim_std_prod = (sim_data * exp_std**2)
        exp_grad = (-1.0/self._inv_std_sum) * np.sum((sim_jac.T / sim_std_prod), axis=1)
        self._sf_gradient = self._sf * exp_grad

    def calc_sf_prior_gradient(self):
        """
        prior penalty is: ((log(B(theta)) - log_B_prior) / sigma_b_prior)**2

        derive (log(B(theta)) = 1/B(theta) * dB/dtheta
        dB/dtheta is the scale factor gradient
        """
        if self.log_prior is None:
            return None
        return self._sf_gradient / self._sf

    def calc_sf_prior_residual(self):
        """
        prior penalty is: ((log(B(theta)) - log_B_prior) / sigma_b_prior)**2
        """
        if self.log_prior is None:
            return None
        return (np.log(self._sf) - self.log_prior) / self.log_prior_sigma

    def calc_scale_factor_entropy(self, sim_data, exp_data, exp_std, temperature=1.0):
        """
        Implementation taken from SloppyCell.  All credit to Sethna group, all mistakes are mine
        """
        if self._sf is None:
            return 0

        sim_dot_exp = np.sum((sim_data * exp_data) / (exp_std ** 2))
        sim_dot_sim = np.sum((sim_data * sim_data) / (exp_std ** 2))

        self._sf = sim_dot_exp / sim_dot_sim
        log_sf = np.log(self._sf)

        integral_args = (sim_dot_sim, sim_dot_exp, self.log_prior, self.log_prior_sigma, temperature,
                         self._sf, log_sf)
        ans, temp = scipy.integrate.quad(_entropy_integrand, -scipy.inf, scipy.inf, args=integral_args, limit=1000)
        entropy = np.log(ans)
        return entropy

    @property
    def sf(self):
        return self._sf

    @property
    def gradient(self):
        return self._sf_gradient.copy()
