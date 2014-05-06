import numpy as np

from measurement import TimecourseMeasurement


class Experiment(object):
    """
    Creates an Experiment.  This is the basic data structure to describe measurements and experimental
    conditions.

    Currently only supports time-course based measurements.

    Attributes
    ----------
    name: string
        The name of the experiment
    exp_data: :class:`~TimecourseMeasurement:SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement`
        A timeseries measurement.
    fixed_parameters: dict, optional
        A dictionary of parameter_name, value pairs which contains parameters which are
        fixed in a particular experiment
        Example:
            fixed_params = {'kon': 0.05, 'koff': 0.013}
    experiment_settings:  dict, optional
        A dictionary of settings upon which parameters that are optimized can vary.\n
        Parameter dependency upon settings is specified in a separate file, allowing us
        to use the same model and experiments, but only vary the dependencies.\n
        Example:
            param_settings = {'decay_rate': 'high'}
        """

    def __init__(self, name, measurements, fixed_parameters=None,
                 experiment_settings=None):

        self.name = name
        self.fixed_parameters = fixed_parameters
        self.settings = {}
        self.initial_conditions = {}

        if experiment_settings is not None:
            for key, value in experiment_settings.items():
                self.settings[key] = value

        self.measurements = []
        if hasattr(measurements, '__iter__'):
            for measurement in measurements:
                self.add_measurement(measurement)

        else:
            self.measurements.append(measurements)  # Make sure these are unique per variable
        self.param_global_vector_idx = None

    def drop_timepoint_zero(self, variable=None):
        """
        Removes all measurements occurring at timepoint zero.

        This is useful because often we don't wish to fit the t0 of our model since
        it does not depend on the parameters.
        """
        for measure_idx in range(len(self.measurements)):
            measurement_variable = self.measurements[measure_idx].variable_name
            if (variable is None) or (measurement_variable == variable):
                self.measurements[measure_idx].drop_timepoint_zero()

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
        for measurement in self.measurements:
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
        for measurement in self.measurements:
            if variable_name == measurement.variable_name:
                return measurement
        raise KeyError('%s not in measurements' %variable_name)

    def add_measurement(self, measurement):
        """
        Adds a measurement to the experiment

        Raises `KeyError` if there is already another timeseries measurement of `variable_name`

        Parameters
        ----------
        measurement: measurement: :class:`~TimecourseMeasurement:SysBio_Modeling.measurement.timecourse_measurement.TimecourseMeasurement`
            A new measurement to add to the experiment
        """
        if len(self.measurements) == 0:
            self.measurements.append(measurement)

        else:
            for existing_measurement in self.measurements:
                if measurement.variable_name == existing_measurement.variable_name:
                    if (type(existing_measurement) and type(measurement)) is TimecourseMeasurement:
                        raise KeyError('%s already has timeseries data associated with this experiment')
                    # Two steady state measurements are OK provided they are wrt different parameters.
            self.measurements.append(measurement)

