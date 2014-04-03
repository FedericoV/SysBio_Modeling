from unittest import TestCase
import numpy as np
from experiment.experiments import Experiment
from nose.tools import raises

__author__ = 'Federico Vaggi'


class TestExperiment(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': np.log(exp_measures)}}
        simple_exp = Experiment('Simple_Experiment', exp_data)
        cls.simple_exp = simple_exp

    @raises(TypeError)
    def test_empty_experiment(self):
        """
        Trying to build an experiment without data should raise an error
        """
        exp_data = None
        no_data_exp = Experiment('Simple_Experiment', exp_data)

    @raises(ValueError)
    def test_wrong_data(self):
        """
        Experiments where the number of timepoints is different from the
        number of measurements should raise an error
        """
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  66.66666667,   77.77777778,   88.88888889])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])

        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': np.log(exp_measures)}}
        no_data_exp = Experiment('Simple_Experiment', exp_data)

    def test_drop_zero_timepoints(self):
        """
        We are dropping the zeroth timepoint
        """
        exp = TestExperiment.simple_exp
        exp.drop_timepoint_zero()
        assert (exp.measurements['Variable_1']['timepoints'][0] == 11.11111111)

    def test_get_unique_timepoints(self):
        """
        Get unique timepoints across two measurements
        """

        exp_timepoints = np.arange(0, 8, 1)
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324])
        exp_timepoints_2 = np.array([0.         ,         3.3,   22.22222222,   33.33333333,
                                    44.44444444,  66.66666667,   77.77777778,   88.88888889])
        exp_data = {'Variable_1': {'timepoints': exp_timepoints, 'value': np.log(exp_measures)},
                    'Variable_3': {'timepoints': exp_timepoints_2, 'value': exp_measures}}
        exp = Experiment('Simple_Experiment', exp_data)
        unique_t = exp.get_unique_timepoints(True)

        sorted_t = np.sort(unique_t)
        assert(np.array_equal(unique_t, sorted_t))  # Check that it is sorted
        assert (len(np.setdiff1d(exp_timepoints, sorted_t)) == 0)   # Check that all elements are there
        assert (len(np.setdiff1d(exp_timepoints_2, sorted_t)) == 0)  # Check that all elements are there
