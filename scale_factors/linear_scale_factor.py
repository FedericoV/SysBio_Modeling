__author__ = 'Federico Vaggi'

from abstract_scale_factor import ScaleFactorABC
import numpy as np
import scipy
import numba
import copy

########################################################################################
# Utility Functions
########################################################################################
@numba.jit
def _entropy_integrand(u, ak, bk, prior_B, sigma_log_B, T, B_best, log_B_best):
    """Copied from SloppyCell"""
    B_centered = np.exp(u) * B_best
    lB = u + log_B_best
    return np.exp(-ak / (2 * T) * (B_centered - B_best) ** 2 - (lB - prior_B) ** 2 / (2 * sigma_log_B ** 2))


def _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, exp_weight=1):
    sim_dot_exp[:] += np.sum(((exp_data/exp_std**2) * sim_data)) * exp_weight
    sim_dot_sim[:] += np.sum(((sim_data/exp_std) * (sim_data/exp_std))) * exp_weight


def _accumulate_scale_factors_jac(exp_data, exp_std, sim_data, model_sens,
                                  sim_dot_exp, sim_dot_sim, sens_dot_exp_data, sens_dot_sim, exp_weight=1):
    sens_dot_exp_data[:] += np.sum(model_sens.T*exp_data / (exp_std**2), axis=1) * exp_weight  # Vector
    sens_dot_sim[:] += np.sum(model_sens.T*sim_data / (exp_std**2), axis=1) * exp_weight  # Vector
    sim_dot_sim[:] += np.sum((sim_data * sim_data) / (exp_std**2)) * exp_weight  # Scalar
    sim_dot_exp[:] += np.sum((sim_data * exp_data) / (exp_std**2)) * exp_weight  # Scalar


def _combine_scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp, scale_jac_out):
    scale_jac_out[:] = (sens_dot_exp_data/sim_dot_sim - 2*sim_dot_exp*sens_dot_sim/sim_dot_sim**2)


class LinearScaleFactor(ScaleFactorABC):

    def __init__(self, log_prior=None, log_prior_sigma=None):
        super(LinearScaleFactor, self).__init__(log_prior, log_prior_sigma)
        self._sf = 1.0

    def update_sf(self, measure_iterator):
        sim_dot_exp = np.zeros((1,), dtype='float64')
        sim_dot_sim = np.zeros((1,), dtype='float64')

        for (measurement, sim, model_sens, exp_weight) in measure_iterator:
            exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()
            sim_data = sim['value']
            _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, 1)

        self._sf = sim_dot_exp / sim_dot_sim

    def update_sf_gradient(self, measure_iterator, n_global_pars):
        """
        Analytically calculates the gradient of the scale factors for each measurement
        """

        scale_factor_gradient = np.zeros((n_global_pars,), dtype='float64')
        sens_dot_exp_data = np.zeros((n_global_pars,), dtype='float64')
        sens_dot_sim = np.zeros((n_global_pars,), dtype='float64')
        sim_dot_sim = np.zeros((1,), dtype='float64')
        sim_dot_exp = np.zeros((1,), dtype='float64')

        for (measurement, sim, model_sens, exp_weight) in measure_iterator:
            exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()
            sim_data = sim['value']
            _accumulate_scale_factors_jac(exp_data, exp_std, sim_data, model_sens, sim_dot_exp, sim_dot_sim,
                                          sens_dot_exp_data, sens_dot_sim, 1)

        _combine_scale_factors(sens_dot_exp_data, sens_dot_sim, sim_dot_sim, sim_dot_exp,
                               scale_factor_gradient)

        self._sf_gradient = scale_factor_gradient

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

    def calc_scale_factor_entropy(self, measure_iterator, temperature=1.0):
        """
        Implementation taken from SloppyCell.  All credit to Sethna group, all mistakes are mine
        """
        if self._sf is None:
            return 0

        sim_dot_exp = np.zeros((1,), dtype='float64')
        sim_dot_sim = np.zeros((1,), dtype='float64')

        for (measurement, sim, model_sens, exp_weight) in measure_iterator:
            exp_data, exp_std, exp_timepoints = measurement.get_nonzero_measurements()
            sim_data = sim['value']
            _accumulate_scale_factors(exp_data, exp_std, sim_data, sim_dot_exp, sim_dot_sim, exp_weight)

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

    def __repr__(self):
        output = "SF value: %.4f\n" % self._sf

        if self.log_prior is not None:
            output += "SF log prior: %.4f\n" % self.log_prior
            output += "SF log prior sigma: %.4f\n" % self.log_prior_sigma

        return output