__author__ = 'Federico Vaggi'

import numpy as np
import numba
from assimulo.problem import Explicit_Problem
from assimulo.solvers import CVode
from assimulo.solvers.sundials import CVodeError
from assimulo.exception import TimeLimitExceeded
from abstract_model import ModelABC


def _make_rhs(odefunc, n_vars):
    yout = np.zeros((n_vars,))

    def assimulo_func_wrapper(t, y, p):
        odefunc(y, t, yout, p)
        return yout
    return assimulo_func_wrapper


class NumbaExplicitProblem(Explicit_Problem):
    def __init__(self, new_rhs, y0):
        Explicit_Problem.__init__(self, y0=y0)
        self.rhs_func = new_rhs

    def rhs(self, t, y, p):
        return self.rhs_func(t, y, p)


class AssimuloCVode(ModelABC):
    def __init__(self, model, n_vars, param_order, model_name="Model",
                 use_jit=True):
        self._unjitted_model = model  # Keep unjitted version just in case
        if use_jit:
            model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(model)
            self._jit_enabled = True
        else:
            self._jit_enabled = False

        super(AssimuloCVode, self).__init__(model, n_vars, param_order, model_name)
        self.param_order = param_order

        rhs = _make_rhs(model, n_vars)
        self.explicit_problem = NumbaExplicitProblem(rhs, y0=np.zeros((n_vars,)))
        self.explicit_sim = self.make_explicit_sim()

    def make_explicit_sim(self):
        explicit_sim = CVode(self.explicit_problem)
        explicit_sim.iter = 'Newton'
        explicit_sim.discr = 'BDF'
        explicit_sim.rtol = 1e-7
        explicit_sim.atol = 1e-7
        explicit_sim.sensmethod = 'SIMULTANEOUS'
        explicit_sim.suppress_sens = True
        explicit_sim.report_continuously = False
        explicit_sim.usesens = False
        explicit_sim.verbosity = 50

        return explicit_sim

    def enable_jit(self):
        if self._jit_enabled:
            print "Model is already JIT'ed using Numba"
        else:
            numba_model = numba.jit("void(f8[:], f8, f8[:], f8[:])")(self._model)
            rhs = _make_rhs(numba_model, self.n_vars)
            explicit_problem = NumbaExplicitProblem(rhs, y0=np.zeros((self.n_vars,)))

            self.explicit_problem = explicit_problem
            self.explicit_sim = self.make_explicit_sim()
            self._model = numba_model
            self._jit_enabled = True

    def calc_jacobian(self, experiment_params, t_sim, init_conditions=None):

        self.explicit_sim.reset()
        self.explicit_sim.report_continuously = True
        self.explicit_sim.usesens = True

        if init_conditions is not None:
            self.explicit_sim.y0 = init_conditions

        self.explicit_sim.p0 = experiment_params
        #self.explicit_sim.pbar = experiment_params

        t_end = t_sim[-1]

        self.explicit_sim.simulate(t_end, ncp_list=t_sim)

        jac = np.array(self.assimulo_sim.p_sol)
        jac = jac.reshape(jac.shape[0], jac.shape[1])

        return jac

    def simulate_experiment(self, experiment_params, t_sim, init_conditions=None):

        self.explicit_problem.p0 = experiment_params
        self.explicit_sim = self.make_explicit_sim()
        self.explicit_sim.y0 = init_conditions

        t_end = t_sim[-1]

        assi_t, assi_y = self.explicit_sim.simulate(t_end, ncp_list=t_sim)

        return assi_y
