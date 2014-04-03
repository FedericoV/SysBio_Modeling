from abc import ABCMeta, abstractmethod, abstractproperty


class ModelABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, model, n_vars):
        self.model = model
        self._n_vars = n_vars

    @abstractmethod
    def simulate_experiment(self, global_param_vector, t_sim, experiment):
        return NotImplementedError

    def get_n_vars(self):
        return self._n_vars

    n_vars = abstractproperty(get_n_vars)

    def calc_jacobian(self):
        raise NotImplementedError
