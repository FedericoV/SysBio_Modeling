import numpy as np


class Experiment(object):
    """
    Creates an Experiment.  This is the basic data structure to describe measurements and experimental
    conditions.

    Currently only supports time-course based measurements.

    :param name: The name of the experiment
    :type: string
    :param measurements: A timeseries measurement or a list of timeseries measurements associated with the experiment
    :type list[SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement] |
        SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement
    :param fixed_parameters: A dictionary of parameter_name, value pairs which contains parameters which are
        fixed in a particular experiment.
        Example:
            fixed_params = {'kon': 0.05, 'koff': 0.013}
    :type dict, optional
    :param experiment_settings: A dictionary of settings upon which parameters that are optimized can vary.\n
        Parameter dependency upon settings is specified in a separate file, allowing us
        to use the same model and experiments, but only vary the dependencies.
        Example:
            param_settings = {'decay_rate': 'high'}
    :type: dict, optional

        """

    def __init__(self, name, measurements, fixed_parameters=None,
                 experiment_settings=None):

        if name[0].isalnum():
            self.name = name
        else:
            raise ValueError("Experiment names must start with a letter or number")

        self.fixed_parameters = fixed_parameters
        self.settings = {}
        self.initial_conditions = {}

        if experiment_settings is not None:
            for key, value in experiment_settings.items():
                self.settings[key] = value

        self._measurements = []
        if hasattr(measurements, '__iter__'):
            for measurement in measurements:
                self.add_measurement(measurement)

        else:
            self.add_measurement(measurements)
        self.param_global_vector_idx = None

    @property
    def measurements(self):
        return self._measurements

    def drop_timepoint_zero(self, variable=None):
        """
        Removes all measurements occurring at timepoint zero.

        This is useful because often we don't wish to fit the t0 of our model since
        it does not depend on the parameters.
        """
        for measure_idx in range(len(self._measurements)):
            measurement_variable = self._measurements[measure_idx].variable_name
            if (variable is None) or (measurement_variable == variable):
                self._measurements[measure_idx].drop_timepoint_zero()

    def get_unique_timepoints(self, include_zero=False):
        """
        Returns the union of all timepoints across all the measurements in the experiment.

        Parameters
        ----------
        include_zero: bool, optional
            Whether to return the zero timepoint measurements.

        Returns
        -------
        unique_timepoints: :class:`~numpy:numpy.ndarray`
            All the timepoints across all the measurements, sorted.
        """
        all_timepoints = []
        for measurement in self._measurements:
            exp_timepoints = measurement.timepoints
            all_timepoints.append(exp_timepoints)

        unique_timepoints = np.unique(np.concatenate(all_timepoints))
        if not include_zero:
            unique_timepoints = unique_timepoints[unique_timepoints != 0]

        unique_timepoints.sort()

        return unique_timepoints

    def get_variable_measurements(self, variable_name):
        """
        Returns the measurement object associated with a particular variable

        Raises `KeyError` if there is no measurement of `variable_name`

        Parameters
        ----------
        variable_name: string
            The name of the measured variable

        Returns
        -------
        measurement: :class:`~TimecourseMeasurement:SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement`
            A measurement object
        """
        for measurement in self._measurements:
            if variable_name == measurement.variable_name:
                return measurement
        raise KeyError('%s not in measurements' % variable_name)

    def add_measurement(self, measurement):
        """
        Adds a measurement to the experiment

        Raises `KeyError` if there is already another timeseries measurement of `variable_name`

        Parameters
        ----------
        measurement: measurement: :class:`~TimecourseMeasurement:SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement`
            A new measurement to add to the experiment
        """
        if len(self._measurements) == 0:
            self._measurements.append(measurement)

        else:
            for existing_measurement in self._measurements:
                if measurement.variable_name == existing_measurement.variable_name:
                    #if (type(existing_measurement) and type(measurement)) is TimecourseMeasurement:
                    # Two steady state measurements are OK provided they are wrt different parameters.
                    raise KeyError('%s already has timeseries data associated with this experiment' % self.name)
            self._measurements.append(measurement)
        self._measurements.sort(key=lambda x: x.variable_name)
