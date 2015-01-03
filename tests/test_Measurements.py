__author__ = 'Federico Vaggi'

from unittest import TestCase

import numpy as np
from nose.tools import raises

from ..measurement import TimecourseMeasurement


class TestTimecourseMeasurement(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)
        cls.measure = measure

    @raises(ValueError)
    def test_wrong_timepoints(self):
        """
        Experiments where the number of timepoints is different from the
        number of measurements should raise an error
        """
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  66.66666667,   77.77777778,   88.88888889])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])

        measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)

    @raises(ValueError)
    def test_wrong_std(self):
        """
        Experiments where the number of standard deviations is different from the
        number of measurements should raise an error
        """
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        exp_std = np.array([0.5, 0.3])

        measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints, exp_std)

    def test_drop_zero_timepoints(self):
        """
        We are dropping the zeroth timepoint
        """
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)

        measure.drop_timepoint_zero()
        assert (measure.timepoints[0] == 11.11111111)