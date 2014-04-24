__author__ = 'Federico Vaggi'

from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np


class MeasurementABC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, variable_name, measurement_value, measurement_std=None):
        if measurement_std is None:
            measurement_std = np.ones_like(measurement_value)

        if not (len(measurement_value) == len(measurement_std)):
            raise ValueError('Length of Standard Deviation Array Not Equal to Length of Measurements')

        self.variable_name = variable_name
        self.values = measurement_value
        self.std = measurement_std
