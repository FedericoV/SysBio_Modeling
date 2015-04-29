__author__ = 'Federico Vaggi'

from .squared_loss_function import SquareLossFunction


class NormalizedSquareLossFunction(SquareLossFunction):
    """
    Normalized Square Loss Function:

    .. math::

        C(\theta)= 0.5*(\frac{\sum{BX_i - Y_i}}^2{(\sigma_i * Y_i))}^2

    Where:

    `X_i` is f(\theta, i), `Y_i` is a vector of measurements, `\sigma_i` is the standard deviation
    of the measurements, and B is an (optional) scale factor in case the measurements are in arbitrary units.

    The main difference between this loss function and the default SquareLossFunction is that we divide the residuals
    by the measurement mean - which is necessary when we are working with multiple measurements on different scales.
    """

    def residuals(self, simulations, experiment_measures):
        em = experiment_measures.copy()
        self.scale_experiment_measures(em)
        return super(NormalizedSquareLossFunction, self).residuals(simulations, em)

    def update_scale_factors(self, simulations, experiment_measures):
        em = experiment_measures.copy()
        self.scale_experiment_measures(em)
        return super(NormalizedSquareLossFunction, self).update_scale_factors(simulations, em)

    def update_scale_factors_gradient(self, simulations, experiment_measures, simulations_jacobian):
        em = experiment_measures.copy()
        self.scale_experiment_measures(em)
        return super(NormalizedSquareLossFunction, self).update_scale_factors_gradient(simulations, em,
                                                                                       simulations_jacobian)

    @staticmethod
    def scale_experiment_measures(experiment_measures):
        if ("~Prior" in experiment_measures.index.levels[0]) or ("~~SF_Prior" in experiment_measures.index.levels[0]):
            # We use drop to get a view of the dataframe without priors
            no_priors_exp = experiment_measures.drop(["~Prior", "~~SF_Prior"], axis=0, level=0)
            no_priors_exp['std'] *= no_priors_exp['mean']  # Scaling standard deviation by mean to have a relative loss
        else:
            experiment_measures['std'] *= experiment_measures['mean']

