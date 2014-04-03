# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:57:47 2013

@author: Federico Vaggi (vaggi.federico@gmail.com)
"""
from collections import OrderedDict
import numpy as np


class Experiment(object):
    def __init__(self, name, exp_data, fixed_parameters=None,
                 experiment_settings=None):
        """
        Constructs an experiment.
        """

        self.name = name
        self.fixed_parameters = fixed_parameters
        self.settings = {}
        self.measurements = OrderedDict()
        self.initial_conditions = {}

        if experiment_settings is not None:
            for key, value in experiment_settings.items():
                self.settings[key] = value

        for variable in exp_data:
            self.measurements[variable] = {}
            measurements = exp_data[variable]
            timepoints = measurements['timepoints']
            value = measurements['value']

            if 'std_dev' not in measurements:
                std_dev = np.ones_like(value)

            else:
                std_dev = measurements['std_dev']

            if not ((len(timepoints) == len(value)) and (len(timepoints) == len(std_dev))):
                raise ValueError('Number of timepoints does not match number of measures')

            self.measurements[variable]['timepoints'] = timepoints
            self.measurements[variable]['value'] = value
            self.measurements[variable]['std_dev'] = std_dev

        self.param_global_vector_idx = None

    def drop_timepoint_zero(self):
        for measurement in self.measurements:
            timepoints = self.measurements[measurement]['timepoints']
            values = self.measurements[measurement]['value']
            std_dev = self.measurements[measurement]['std_dev']

            values = values[timepoints != 0]
            std_dev = std_dev[timepoints != 0]
            timepoints = timepoints[timepoints != 0]

            self.measurements[measurement]['value'] = values
            self.measurements[measurement]['timepoints'] = timepoints
            self.measurements[measurement]['std_dev'] = std_dev

    def get_unique_timepoints(self, include_zero_timepoints=False):
        all_timepoints = []
        for measurement in self.measurements:
            exp_timepoints = self.measurements[measurement]['timepoints']
            all_timepoints.append(exp_timepoints)

        unique_timepoints = np.unique(np.concatenate(all_timepoints))
        if not include_zero_timepoints:
            unique_timepoints = unique_timepoints[unique_timepoints != 0]

        unique_timepoints.sort()

        return unique_timepoints
