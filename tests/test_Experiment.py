from unittest import TestCase
import numpy as np
from experiment import Experiment
from nose.tools import raises
from measurement import TimecourseMeasurement

__author__ = 'Federico Vaggi'


class TestExperiment(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        simple_measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)

        simple_exp = Experiment('Simple_Experiment', simple_measure)
        cls.simple_exp = simple_exp

    @raises(KeyError)
    def test_get_missing_measurement(self):
        simple_exp = TestExperiment.simple_exp
        simple_exp.get_variable_measurements('Not_There')

    def test_get_unique_timepoints(self):
        """
        Get unique timepoints across two measurements
        """

        exp_timepoints = np.arange(0, 8, 1)
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324])
        exp_timepoints_2 = np.array([0.         ,         3.3,   22.22222222,   33.33333333,
                                    44.44444444,  66.66666667,   77.77777778,   88.88888889])

        measure_1 = TimecourseMeasurement('Variable_1', exp_measures, exp_timepoints)
        measure_2 = TimecourseMeasurement('Variable_3', exp_measures, exp_timepoints_2)
        exper = Experiment('Simple_Experiment', [measure_1, measure_2])
        unique_t = exper.get_unique_timepoints(True)

        sorted_t = np.sort(unique_t)
        assert(np.array_equal(unique_t, sorted_t))  # Check that it is sorted
        assert (len(np.setdiff1d(exp_timepoints, sorted_t)) == 0)   # Check that all elements are there
        assert (len(np.setdiff1d(exp_timepoints_2, sorted_t)) == 0)  # Check that all elements are there

    @raises(KeyError)
    def test_add_measure(self):
        """Fails because there is already another timecourse measurement of Variable_1"""

        exp_timepoints = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                                   44.44444444,  55.55555556,   66.66666667,   77.77777778,
                                   88.88888889,  100.])
        exp_measures = np.array([0.74524402,  1.53583955,  2.52502335,  3.92107899,  4.58210253,
                                 5.45036258,  7.03185055,  7.75907324,  9.30805318,  9.751119])
        measure2 = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)

        TestExperiment.simple_exp.add_measurement(measure2)

    def test_lexsorted(self):
        """Tests that measurements added remain lexsorted"""
        b = np.ones((5,))

        measure_1 = TimecourseMeasurement('aaaa', b, b)
        measure_2 = TimecourseMeasurement('zzaa', b, b)
        measure_3 = TimecourseMeasurement('baaa', b, b)
        measure_4 = TimecourseMeasurement('bcaaa', b, b)
        measure_5 = TimecourseMeasurement('125baaa', b, b)
        measure_6 = TimecourseMeasurement('aaa11', b, b)

        measure_list = [measure_1, measure_2, measure_3, measure_4, measure_5, measure_6]
        measure_name_list = [measure.variable_name for measure in measure_list]
        measure_name_list.sort()

        exper = Experiment('Sorted_Experiment', measure_list)

        for measure, m_name in zip(exper.measurements, measure_name_list):
            assert(measure.variable_name == m_name)






