from abc import ABCMeta, abstractmethod, abstractproperty


class ModelABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, model, n_vars, param_order, model_name):
        self._model = model
        self._n_vars = n_vars
        self.model_name = model_name
        self.param_order = param_order

    @abstractmethod
    def simulate_experiment(self, project_param_vector, t_sim, experiment):
        return NotImplementedError

    def get_n_vars(self):
        return self._n_vars

    n_vars = property(get_n_vars)

    def calc_jacobian(self):
        raise NotImplementedError
