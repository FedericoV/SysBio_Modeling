__author__ = 'Federico Vaggi'

from .abstract_measurement import MeasurementABC


class TimecourseMeasurement(MeasurementABC):

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