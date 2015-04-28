__author__ = 'Federico Vaggi'

from .squared_loss_function import SquareLossFunction


class NormalizedSquareLossFunction(SquareLossFunction):

    def residuals(self, simulations, experiment_measures):
        experiment_measures = experiment_measures.copy()
        self.scale_experiment_measures(experiment_measures)
        return super(NormalizedSquareLossFunction, self).residuals(simulations, experiment_measures)

    def update_scale_factors(self, simulations, experiment_measures):
        experiment_measures = experiment_measures.copy()
        self.scale_experiment_measures(experiment_measures)
        return super(NormalizedSquareLossFunction, self).update_scale_factors(simulations, experiment_measures)

    def update_scale_factors_gradient(self, simulations, experiment_measures, simulations_jacobian):
        experiment_measures = experiment_measures.copy()
        self.scale_experiment_measures(experiment_measures)
        return super(NormalizedSquareLossFunction, self).update_scale_gradient(self, simulations, experiment_measures,
                                                                        simulations_jacobian)

    @staticmethod
    def scale_experiment_measures(experiment_measures):
        if ("~Prior" in experiment_measures.index.levels[0]) or ("~~SF_Prior" in experiment_measures.index.levels[0]):
            # We use drop to get a view of the dataframe without priors
            no_priors_exp = experiment_measures.drop(["~Prior", "~~SF_Prior"], axis=0, level=0)
            no_priors_exp['std'] *= no_priors_exp['mean']  # Scaling standard deviation by mean to have a relative loss
        else:
            experiment_measures['std'] *= experiment_measures['mean']

