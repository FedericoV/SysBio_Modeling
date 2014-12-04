from unittest import TestCase
from nose.tools import raises

import numpy as np
from statsmodels.tools.numdiff import approx_fprime

from ..experiment import Experiment
from ..measurement import TimecourseMeasurement
from ..project import Project
from ..model import ode_model

from test_utils.simple_model_settings import settings as experiment_settings
from test_utils.jittable_model import model
from test_utils.sens_jittable_model import sens_model


__author__ = 'Federico Vaggi'


def _simple_model_analytical_jac(k_deg, k_synt, t):
    k_synt_jac = k_synt * (1 / k_deg - np.exp(-k_deg * t) / k_deg)
    k_deg_jac = k_deg * (
        k_synt * t * np.exp(-k_deg * t) / k_deg - k_synt / k_deg ** 2 + k_synt * np.exp(-k_deg * t) / k_deg ** 2)
    return np.vstack((k_deg_jac, k_synt_jac))


class TestProject(TestCase):
    @classmethod
    def setUpClass(cls):
        exp_timepoints = np.array([0., 11.11111111, 22.22222222, 33.33333333,
                                   44.44444444, 55.55555556, 66.66666667, 77.77777778,
                                   88.88888889, 100.])
        low_deg_measures = np.array([0., 0.11049608, 0.21977125, 0.32783897, 0.43471258, 0.54995561,
                                     0.65437492, 0.75764044, 0.85976492, 0.9516258]) * 3.75
        measurement_1 = TimecourseMeasurement('Variable_1', low_deg_measures, exp_timepoints)
        exp_settings = {'Deg_Rate': 'Low'}
        low_deg = Experiment('Low_Deg_Exp', measurement_1, experiment_settings=exp_settings)

        exp_timepoints_2 = np.array([5.05050505, 9.09090909, 12.12121212, 16.16161616, 19.19191919,
                                     23.23232323, 26.26262626, 30.3030303, 33.33333333, 37.37373737,
                                     40.4040404, 44.44444444, 47.47474747, 50.50505051, 54.54545455,
                                     57.57575758, 61.61616162, 64.64646465, 68.68686869, 71.71717172,
                                     75.75757576, 78.78787879, 82.82828283, 85.85858586, 89.8989899,
                                     92.92929293])

        high_deg_measures = np.array([0.04925086, 0.08689927, 0.11415396, 0.14923229, 0.17462643, 0.20731014,
                                      0.23097075, 0.26142329, 0.28346868, 0.31184237, 0.33238284, 0.3588196,
                                      0.37795787, 0.39652489, 0.42042171, 0.43772125, 0.45998675, 0.47610533,
                                      0.49685087, 0.51186911, 0.53119845, 0.54519147, 0.56320128, 0.57623905,
                                      0.59301942, 0.60516718]) * 3.75

        measurement_2 = TimecourseMeasurement('Variable_1', high_deg_measures, exp_timepoints_2)
        exp_settings_2 = {'Deg_Rate': 'High'}
        high_deg = Experiment('High_Deg_Exp', measurement_2, experiment_settings=exp_settings_2)
        experiments = [low_deg, high_deg]

        # Model
        ordered_params = ['k_deg', 'k_synt']
        n_vars = 1
        cls.ode_model = ode_model.OdeModel(model, sens_model, n_vars, ordered_params)

        measurement_to_model_map = {'Variable_1': ('direct', 0)}
        proj = Project(cls.ode_model, experiments, experiment_settings, measurement_to_model_map)
        cls.proj = proj

        project_param_vector = np.zeros((3,))
        low_deg_idx = proj.get_param_index('Group_1', ('Low',))
        high_deg_idx = proj.get_param_index('Group_1', ('High',))
        synt_idx = proj.get_param_index('k_synt', 'Global')

        project_param_vector[high_deg_idx] = 0.01
        project_param_vector[low_deg_idx] = 0.001
        project_param_vector[synt_idx] = 0.01

        log_project_param_vector = np.log(project_param_vector)
        cls.log_project_param_vector = log_project_param_vector

    def test__project_initialization(self):
        proj = TestProject.proj
        _project_param_idx = proj._project_param_idx
        k_synt_settings = proj.get_param_index('k_synt')
        assert (k_synt_settings.keys() == ['Global'])
        k_deg_settings = proj.get_param_index('Group_1')
        assert (len(k_deg_settings) == 2)  # Two settings for k_deg

        for exp in proj.experiments:
            setting = exp.settings['Deg_Rate']
            assert (_project_param_idx['Group_1'][(setting,)] == exp.param_global_vector_idx['k_deg'])
            # Check parameters are set properly

        assert (proj.n_project_residuals == 35)

    def test_set_measurement_idx(self):
        proj = TestProject.proj
        assert (0 in proj._measurements_idx['Variable_1'])
        assert (1 in proj._measurements_idx['Variable_1'])

    def test_sim_experiments(self):
        proj = TestProject.proj
        log_project_param_vector = TestProject.log_project_param_vector
        proj(log_project_param_vector)
        assert (len(proj._all_sims) == 2)

        for exp_idx, experiment in enumerate(proj.experiments):
            measurement = experiment.get_variable_measurements('Variable_1')
            exp_data, _, _ = measurement.get_nonzero_measurements()
            exp_data /= 3.75
            sim = proj._all_sims[exp_idx]['Variable_1']
            assert (np.allclose(exp_data, sim['value'], rtol=0.05))

        assert (np.allclose(proj._scale_factors['Variable_1'].sf, 3.75, rtol=0.05))
        # Excellent

        for exp_idx, res_block in enumerate(proj._all_residuals):
            experiment = proj.get_experiment(exp_idx)
            measurement = experiment.get_variable_measurements('Variable_1')
            exp_data, _, _ = measurement.get_nonzero_measurements()
            total_measurement = np.sum(exp_data)
            total_res = np.sum(res_block['Variable_1'])
            assert (total_measurement * 0.01 > total_res)

    def test_model_jacobian(self):
        proj = TestProject.proj
        log_project_param_vector = TestProject.log_project_param_vector
        project_param_vector = np.exp(log_project_param_vector)
        proj.calc_project_jacobian(log_project_param_vector)

        for exp_idx, jac_block in enumerate(proj._model_jacobian):
            experiment = proj.get_experiment(exp_idx)
            k_synt_idx = experiment.param_global_vector_idx['k_synt']
            k_deg_idx = experiment.param_global_vector_idx['k_deg']
            k_synt = project_param_vector[k_synt_idx]
            k_deg = project_param_vector[k_deg_idx]
            exp_t = experiment.get_variable_measurements('Variable_1').timepoints
            exp_t = exp_t[exp_t != 0]

            anal_jac = _simple_model_analytical_jac(k_deg, k_synt, exp_t)
            k_synt_analytical_jac = anal_jac[1, :]
            k_deg_analytical_jac = anal_jac[0, :]

            k_synt_sensitivity_jac = jac_block['Variable_1'][:, k_synt_idx]
            k_deg_sensitivity_jac = jac_block['Variable_1'][:, k_deg_idx]

            assert np.allclose(k_synt_analytical_jac, k_synt_sensitivity_jac, rtol=0.05)
            assert np.allclose(k_deg_analytical_jac, k_deg_sensitivity_jac, rtol=0.05)

    def test_scale_factor_jacobian(self):
        proj = TestProject.proj
        log_project_param_vector = TestProject.log_project_param_vector
        proj.calc_project_jacobian(log_project_param_vector)
        _scale_factors_gradient = proj._scale_factors['Variable_1'].gradient

        def get_scale_factors(x):
            proj(x)
            return proj._scale_factors['Variable_1'].sf

        num_scale_factors = approx_fprime(log_project_param_vector, get_scale_factors, centered=True)
        assert np.allclose(_scale_factors_gradient, num_scale_factors, rtol=0.01)

    def test_calc_project_jacobian(self):
        # Known test failure.  Most likely due to numerical failures in scaling factor.
        proj = TestProject.proj
        project_param_vector = np.zeros((3,))
        low_deg_idx = proj._project_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj._project_param_idx['Group_1'][('High',)]
        synt_idx = proj._project_param_idx['k_synt']['Global']

        project_param_vector[high_deg_idx] = 0.01
        project_param_vector[low_deg_idx] = 0.001
        project_param_vector[synt_idx] = 0.01
        log_project_param_vector = np.log(project_param_vector)

        sens_jacobian = proj.calc_project_jacobian(log_project_param_vector)

        def get_scaled_sims(x):
            proj(x)
            sims = []
            for sim in proj._all_sims:
                exp_sim = sim['Variable_1']['value']
                sims.extend(exp_sim.tolist())
            sims = np.array(sims)
            scale = proj._scale_factors['Variable_1'].sf
            return sims * scale

        num_global_jac = approx_fprime(log_project_param_vector, get_scaled_sims, centered=True)
        assert (np.sum((num_global_jac - sens_jacobian) ** 2) < 1e-8)

        project_param_vector[high_deg_idx] = 0.02
        project_param_vector[low_deg_idx] = 0.003
        project_param_vector[synt_idx] = 0.05
        log_project_param_vector = np.log(project_param_vector)

        sens_rss_grad = proj.calc_rss_gradient(log_project_param_vector)
        num_rss_jac = approx_fprime(log_project_param_vector, proj.calc_sum_square_residuals, centered=True)
        assert np.allclose(sens_rss_grad, num_rss_jac, atol=0.000001)

    @raises(AssertionError)
    def test_scale_factors_change(self):
        proj = TestProject.proj
        project_param_vector = np.zeros((3,))
        low_deg_idx = proj._project_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj._project_param_idx['Group_1'][('High',)]
        synt_idx = proj._project_param_idx['k_synt']['Global']

        project_param_vector[high_deg_idx] = 0.01
        project_param_vector[low_deg_idx] = 0.001
        project_param_vector[synt_idx] = 0.01

        log_project_param_vector = np.log(project_param_vector)
        mod_param_vector = np.copy(log_project_param_vector)
        mod_param_vector[0] += 0.2

        proj(log_project_param_vector)
        old_scale_factor = proj._scale_factors['Variable_1'].sf
        # Changing param vector:
        proj(mod_param_vector)
        new_scale_factor = proj._scale_factors['Variable_1'].sf

        assert np.allclose(old_scale_factor, new_scale_factor, rtol=0.0001)

    def test_optimization(self):
        from leastsq_mod import leastsq as geo_leastsq
        # Known test failure.  Most likely due to numerical failures in scaling factor.
        proj = TestProject.proj

        base_guess = np.log(np.ones((3,)) * 0.01)

        out = geo_leastsq(proj, base_guess, Dfun=proj.calc_project_jacobian)

    @raises(KeyError)
    def test_remove_absent_experiment(self):
        proj = TestProject.proj
        proj.remove_experiments_by_settings({'Absent:', 5})

    def test_add_experiment(self):
        #############################################################################################
        # Set Up
        #############################################################################################
        proj = TestProject.proj

        exp_timepoints = np.array([0., 11.11111111, 22.22222222, 33.33333333,
                                   44.44444444, 55.55555556, 66.66666667, 77.77777778,
                                   88.88888889, 100.])
        exp_measures = np.array([0.74524402, 1.53583955, 2.52502335, 3.92107899, 4.58210253,
                                 5.45036258, 7.03185055, 7.75907324, 9.30805318, 9.751119])
        simple_measure = TimecourseMeasurement('Variable_1', np.log(exp_measures), exp_timepoints)

        exp_settings_3 = {'Deg_Rate': 'Very High'}
        simple_exp = Experiment('Simple_Experiment', simple_measure, experiment_settings=exp_settings_3)
        proj.add_experiment(simple_exp)
        ###############################################################################################

        # Trying to add the experiment
        present = 0
        for experiment in proj.experiments:
            if experiment.name == 'Simple_Experiment':
                present = 1

        if present == 0:
            raise AssertionError('Unable to add experiment to project')

        # Trying to remove it now
        proj.remove_experiments_by_settings({'Deg_Rate': 'Very High'})
        present = 0
        for experiment in proj.experiments:
            if experiment.name == 'Simple_Experiment':
                present = 1

        n_experiments = len(list(proj.experiments))
        n_experiment_weights = len(proj.experiments_weights)
        if n_experiments != n_experiment_weights:
            raise AssertionError('Weight vector not updated properly')
        if present == 1:
            raise AssertionError('Unable to remove experiment from project')

    def test_experiment_settings(self):
        raise AssertionError('Not Implemented Yet')

    def test_sum_variables(self):
        #############################################################################################
        # Set Up
        #############################################################################################
        from scipy.integrate import odeint
        from test_utils.jittable_mm_model import model, ordered_params
        from test_utils.sens_jittable_mm_model import sens_model
        from test_utils.michelis_menten_model import michelis_menten
        from leastsq_mod import leastsq as geo_leastsq
        import nlopt
        import matplotlib.pyplot as plt

        init_conditions = [0.0, 0]
        vmax = 1e-3
        km = 0.001
        k_synt_s = 0.01
        k_deg_s = 0.01
        k_deg_p = 0.001
        param_vector = np.array([vmax, km, k_synt_s, k_deg_s, k_deg_p])
        # Wiki http://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics

        t_sim = np.linspace(0, 100, 20)
        sim = odeint(michelis_menten, init_conditions, t_sim, args=(param_vector,))

        #  Here we test fitting the sum of S and P
        total_measure = TimecourseMeasurement('Total', np.sum(sim, axis=1), t_sim)
        single_experiment = Experiment('Standard', total_measure)

        mm_model = ode_model.OdeModel(model, sens_model, 2, ordered_params)
        measurement_to_model_map = {'Total': ('sum', [0, 1])}
        proj = Project(mm_model, [single_experiment], {}, measurement_to_model_map)
        proj.use_scale_factors['Total'] = False

        param_dict = {'vmax': {'Global': vmax}, 'km': {'Global': km}, 'k_synt_s': {'Global': k_synt_s},
                      'k_deg_s': {'Global': k_deg_s}, 'k_deg_p': {'Global': k_deg_p}}
        sorted_param_vector = proj.project_param_dict_to_vect(param_dict)
        log_sorted_param_vector = np.log(sorted_param_vector)
        #############################################################################################

        """
        Testing since we didn't specify any parameter settings, all parameters got set as global
        """
        for param in proj._project_param_idx:
            setting = proj._project_param_idx[param].keys()[0]
            assert (setting == 'Global')
        #############################################################################################
        #############################################################################################

        """
        Testing to see that the residuals of the simulation, when using the same parameters of the ODE
        model used to generate it, are close to zero.
        """
        residuals = proj(log_sorted_param_vector)
        assert np.allclose(residuals, np.zeros_like(residuals), atol=0.001)
        #############################################################################################

        """
        The finite differences gradient of the residual sum of squares is similar to the gradient calculated
        using the sensitivity equations
        """
        sens_rss_grad = proj.calc_rss_gradient(log_sorted_param_vector)
        num_rss_grad = approx_fprime(log_sorted_param_vector, proj.calc_sum_square_residuals, centered=True)
        assert np.allclose(sens_rss_grad, num_rss_grad, atol=0.00001)
        #############################################################################################

        """
        The finite differences jacobian of the residuals is similar to the jacobian calculated
        using sensitivity equations.  Note, because of numerical limitations of finite differences and ODE,
        this is only approximately true
        """

        def calc_all_sims(pars):
            proj(pars)
            return proj._all_sims[0]['Total']['value']

        num_jac = approx_fprime(log_sorted_param_vector, calc_all_sims, centered=True)
        sens_jac = proj.calc_project_jacobian(log_sorted_param_vector)
        assert np.allclose(sens_jac, num_jac, atol=0.00001)
        #############################################################################################

        fit_params = geo_leastsq(proj, np.zeros((5,)), jacobian=proj.calc_project_jacobian,
                                 tols=[1e-3, -1.49012e-06, -1.49012e-06, -1.49012e-06, -1.49012e-06, -1.49012e-06,
                                       -1.49012e-06, -1e3])

        proj.calc_sum_square_residuals(fit_params)
        fit_sim = proj._all_sims[0]['Total']['value']
        fit_t = proj._all_sims[0]['Total']['timepoints']

        # Plotting
        plt.plot(fit_t, fit_sim, 'r-')
        plt.plot(t_sim, np.sum(sim, axis=1), 'ro')
        plt.show()
        #############################################################################################

        # try fitting S & P separately
        substrate_measure = TimecourseMeasurement('Substrate', sim[:, 0], t_sim)
        product_measure = TimecourseMeasurement('Product', sim[:, 1], t_sim)

        substrate_experiment = Experiment('Substrate Experiment', [substrate_measure, product_measure])
        measurement_to_model_map = {'Substrate': ('direct', 0), 'Product': ('direct', 1)}

        mm_model = ode_model.OdeModel(model, sens_model, 2, ordered_params)
        sf_groups = [frozenset(['Substrate', 'Product'])]
        proj = Project(mm_model, [substrate_experiment], {}, measurement_to_model_map, sf_groups=sf_groups)

        proj.use_parameter_priors = True
        proj.set_parameter_log_prior('k_synt_s', 'Global', np.log(0.01), 0.5)
        proj.set_scale_factor_log_prior(frozenset(['Substrate', 'Product']), np.log(1.0), 0.1)
        # Strong prior around 1

        # NLopt Optimization
        random_params = np.random.randn(5)
        opt = nlopt.opt(nlopt.GN_CRS2_LM, proj.n_project_params)
        high_bounds = np.ones_like(random_params) * 8
        low_bounds = np.ones_like(random_params) * -8
        opt.set_upper_bounds(high_bounds)
        opt.set_lower_bounds(low_bounds)
        opt.set_min_objective(proj.nlopt_fcn)
        opt.set_maxtime(5)
        nlopt_params = opt.optimize(random_params)

        # Getting simulated values out.
        residuals = proj(nlopt_params)
        sub_sim = proj._all_sims[0]['Substrate']['value']
        sub_t = proj._all_sims[0]['Substrate']['timepoints']

        prod_sim = proj._all_sims[0]['Product']['value']
        prod_t = proj._all_sims[0]['Product']['timepoints']

        assert (proj._scale_factors['Product'] is proj._scale_factors['Substrate'])
        # They are the same scale factor

        sf = proj._scale_factors['Product'].sf

        plt.plot(sub_t, sub_sim*sf, 'r-')
        plt.plot(t_sim, sim[:, 0], 'ro')
        plt.plot(prod_t, prod_sim*sf, 'b-')
        plt.plot(t_sim, sim[:, 1], 'bo')
        plt.show()

        param_priors = proj.get_parameter_priors()

        #############################################################################################
        n_param_priors = 0
        for p in param_priors:
            for _ in param_priors[p]:
                n_param_priors += 1
        assert (n_param_priors == 1)

        #############################################################################################
        jac = proj.calc_project_jacobian(random_params)
        assert (residuals.shape[0] == jac.shape[0])
