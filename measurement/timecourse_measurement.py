__author__ = 'Federico Vaggi'

from .abstract_measurement import MeasurementABC


class TimecourseMeasurement(MeasurementABC):
    """
    A series of measured values, with their associated timepoints and standard deviations (optimal).

    :param variable_name: The name of the measured variable
    :type: string
    :param measurement_value: An (n,) dimensional array containing measurements of the variable_name
    :type: numpy.array
    :param measurement_time: An (n,) dimensional array containing the times at which measurements were carried out
    :type: numpy.array
    :type: measurement_std: An (n,) dimensional array indicating the uncertanties in the measurements (optional)
    :type: numpy.array

    """
    def __init__(self, variable_name, measurement_value, measurement_time, measurement_std=None):
        super(TimecourseMeasurement, self).__init__(variable_name, measurement_value, measurement_std)
        if not (len(measurement_value) == len(measurement_time)):
            raise ValueError('Length of Standard Deviation Array Not Equal to Length of Timepoints')
        self.timepoints = measurement_time

    def drop_timepoint_zero(self):
        self.values = self.values[self.timepoints != 0]
        self.std = self.std[self.timepoints != 0]
        self.timepoints = self.timepoints[self.timepoints != 0]

    def get_nonzero_measurements(self):
        values = self.values[self.timepoints != 0]
        std = self.std[self.timepoints != 0]
        timepoints = self.timepoints[self.timepoints != 0]
        return values, std, timepoints

    def plot_measurement(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if len(kwargs) == 0:
            kwargs = {'marker': 'o', 'linestyle': '--'}
        ax.errorbar(self.timepoints, self.values, self.std, **kwargs)
        return ax
