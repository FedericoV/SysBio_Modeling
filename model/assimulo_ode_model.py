__author__ = 'Federico Vaggi'

from ode_model import OdeModel


class AssimuloModel(OdeModel):
    def __init__(self, model, n_vars, param_order):
        super(OdeModel, self).__init__(model, n_vars)
        self.param_order = param_order

### Have to put all this in somehow:

import numpy as np
from assimulo.problem import Explicit_Problem
from assimulo.solvers.sundials import CVodeError
from assimulo.exception import TimeLimitExceeded


class p53_assimulo_sim(Explicit_Problem):
    def __init__(self, y0, p0, p53_exp, odeint_fcn=None):

        Explicit_Problem.__init__(self, y0=y0, p0=p0)

        self.conditions = p53_exp.conditions
        self.set_param_order(p53_exp)

        self.odeint_fcn = odeint_fcn
        # Has to follow the Assimulo model signature

    def set_param_order(self, p53_exp):
        param_global_vector_idx = p53_exp.param_global_vector_idx

        self.param_order = {p_name: p_idx_local for p_idx_local, p_name,
                            in enumerate(param_global_vector_idx)}

    def get_param_vector_idx(self, p53_exp):
        param_global_vector_idx = p53_exp.param_global_vector_idx

        p_idx = np.empty(len(param_global_vector_idx))
        for p_name, p_idx_local in self.param_order.items():
            # Only works because param_global_vector_idx is an OrderedDict
            p_idx_global = param_global_vector_idx[p_name]
            p_idx[p_idx_local] = p_idx_global
        return p_idx

    def get_param_vector(self, p53_exp, project_param_vector):

        p_idx = self.get_param_vector_idx(p53_exp)
        p = []
        for model_idx in p_idx:
            p.append(project_param_vector[model_idx])
        p = np.array(p)
        return p

    def rhs(self, t, y, p):
        '''
        Wraps a SciPy odeint function into a format usable by Assimulo.

        Assimulo Explicit_Problem have a .rhs method with a
        signature of (t, y, p) which is called by the the CVode integrator.
        We create a wrapper around a function which has the SciPy odeint
        call function signature (y, t, *args)

        Parameters
        -----------
        odeint_fcn: SciPy odeint function
            A function that can be integrated by SciPy odeint,
            whose signature is y = f(y, t, *args).  We assume that the
            *args field is used to pass the parameters to the function.

        param_global_vector_idx: OrderedDict
            An ordered dictionary that, for each experiment, stores
            the information about where in the global parameter vector
            is the value of a particular parameter.
        '''

        args = (self.param_order, self.conditions, p)
        return self.odeint_fcn(y, t, *args)


def make_assimulo_fcn(assi_sim, experiments, n_vars=6, t_max=32400,
                      verbose=True, std_normalization=False,
                      mean_normalization=False, return_grad_as_tuple=False,
                      scaled_model=False):
    def assi_fitting_fcn(project_param_vector, grad=None):
        residuals = np.array([])

        if return_grad_as_tuple:
            grad = np.zeros_like(project_param_vector)
        if grad.size > 0:
            grad[:] = 0.0

        total_exp_points = 0

        for p53_exp in experiments:
            init_conditions = np.zeros((n_vars,))

            try:
                t_sim, y_sim, exp_sens = p53_exp.simulate_assimulo_model(
                    assi_sim, project_param_vector, init_conditions,
                    scaled_model=scaled_model)

                exp_res = p53_exp.residuals(t_sim, y_sim[:, -1],
                                            std_normalization=std_normalization,
                                            mean_normalization=mean_normalization)

                if grad.size > 0:
                    grad_t = (exp_res * exp_sens.T)
                    grad[:] += np.sum(grad_t, axis=1)

                total_exp_points += len(exp_res)

            except (CVodeError, TimeLimitExceeded) as inst:
                exp_res = 100000  # Large arbitrary value
                print p53_exp.name, inst

            residuals = np.hstack([residuals, exp_res])

        grad[:] /= total_exp_points
        rss = np.sum(residuals ** 2)  # L2 loss

        if verbose:
            print "RSS: %s" % rss
            print "Gradient: %s" % grad
            print "Parameter Vector: %s" % project_param_vector
            print "\n\n"
        if return_grad_as_tuple:
            return rss, grad
        else:
            return rss

    return assi_fitting_fcn


def simulate_assimulo_model(self, assi_sim, project_param_vector,
                            init_conditions=np.zeros(6, ),
                            t_sim=None, calculate_sensitivity=True,
                            scaled_model=False):
    '''t_sim there for compatibility, doesn't do anything'''

    # Sets empty, gal concentration, etc.
    assi_sim.reset()
    assi_sim.problem.conditions = self.conditions
    p = assi_sim.problem.get_param_vector(self, project_param_vector)

    assi_sim.p = p
    assi_sim.pbar = np.ones_like(p)

    endpoint = self.timepoints[-1]

    if scaled_model:
        k_deg_p53 = np.log(2) / (3600 * 6)
        RE_total = 0.0005725
        endpoint = endpoint * k_deg_p53

    if not calculate_sensitivity:
        (t_sim, assi_y) = assi_sim.simulate(endpoint)
        return (t_sim, assi_y)

    t_sim, assi_y = assi_sim.simulate(endpoint, ncp=1000)

    if scaled_model:
        t_sim = t_sim / k_deg_p53
        assi_y = assi_y * RE_total

    t_idx = np.searchsorted(t_sim, self.timepoints[self.timepoints != 0])
    # We ignore sensitivity for point 0

    glob_sens = np.zeros((len(t_idx), len(project_param_vector)))

    param_sens = np.array(assi_sim.p_sol)
    # sensitivity is with respect to the parameters INSIDE the model.
    # A parameter in the project_param_vector can be used twice inside
    # the model - so need to average out the two sensitivities.
    p_idx = assi_sim.problem.get_param_vector_idx(self)
    # p_idx is a vector that tells you where the ith parameter of
    # the model is in the project_parameter_vector - so p[i] = j
    # means that the ith model parameter is in the jth position
    # of project_param_vector.

    for l_idx, g_idx in enumerate(p_idx):
        glob_sens[:, g_idx] += param_sens[l_idx, t_idx, -1]

    return (t_sim, assi_y, glob_sens)
