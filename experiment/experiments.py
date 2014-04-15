from collections import OrderedDict

import numpy as np


class Experiment(object):
    """
    Creates an Experiment.  This is the basic data structure to describe measurements and experimental
    conditions.

    Currently only supports time-course based measurements.

    Attributes
    ----------
    name: string
        The name of the experiment
    exp_data: dict
        A dictionary, containing measurement(s) for multiple species
        Example:
            exp_data['Species_1'] = {'value': 0, 1, 2, 'timepoints': 0, 5, 10}
    fixed_parameters: dict, optional
        A dictionary of parameter_name, value pairs which contains parameters which are
        fixed in a particular experiment
        Example:
            fixed_params = {'kon': 0.05, 'koff': 0.013}
    experiment_settings:  dict, optional
        A dictionary of settings upon which parameters that are optimized
        can vary.  Parameter dependency upon settings is specified in a separate file, allowing us to use the same
        model and experiments, but only vary the dependencies.
        Example:
            param_settings = {'decay_rate': 'high'}
        """

    def __init__(self, name, exp_data, fixed_parameters=None,
                 experiment_settings=None):

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
        """
        Removes all measurements occurring at timepoint zero.

        This is useful because often we don't wish to fit the t0 of our model since
        it does not depend on the parameters.
        """
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
        """
        Returns the union of all timepoints across all the measurements in the experiment.

        Parameters
        ----------
        include_zero_timepoints: bool, optional
            Whether to return the zero timepoint measurements.

        Returns
        -------
        unique_timepoints: :class:`~numpy:numpy.ndarray`
            All the timepoints across all the measurements, sorted.
        """
        all_timepoints = []
        for measurement in self.measurements:
            exp_timepoints = self.measurements[measurement]['timepoints']
            all_timepoints.append(exp_timepoints)

        unique_timepoints = np.unique(np.concatenate(all_timepoints))
        if not include_zero_timepoints:
            unique_timepoints = unique_timepoints[unique_timepoints != 0]

        unique_timepoints.sort()

        return unique_timepoints
