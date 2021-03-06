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
        proj = Project(cls.ode_model, experiments, experiment_settings, measurement_to_model_map,
                       sf_groups=[frozenset(['Variable_1'])])
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
        k_synt_settings = proj.get_param_index('k_synt')
        assert (k_synt_settings.keys() == ['Global'])
        # Check that k_synthesis is set as a global parameter

        k_deg_settings = proj.get_param_index('Group_1')
        assert (len(k_deg_settings) == 2)
        # Two settings for k_deg

        _project_param_idx = proj.project_param_idx
        for exp in proj.experiments:
            setting = exp.settings['Deg_Rate']
            assert (_project_param_idx['Group_1'][(setting,)] == exp.param_global_vector_idx['k_deg'])
            # Check parameters are set properly

        assert (proj.n_project_residuals == 35)

    def test_sim_experiments(self):
        proj = TestProject.proj
        log_project_param_vector = TestProject.log_project_param_vector
        proj.residuals(log_project_param_vector)

        # Count how many distinct experiments we have simulated:
        sim_df = proj.get_simulations()
        simulated_experiments = sim_df.index.levels[0].unique()
        assert (len(simulated_experiments) == 2)

        for exp_idx, experiment in enumerate(proj.experiments):
            measurement = experiment.get_variable_measurements('Variable_1')
            exp_data, _, _ = measurement.get_nonzero_measurements()
            exp_data /= 3.75
            sim_df = proj.get_simulations()
            sim = sim_df.loc[(experiment.name, 'Variable_1'), :].values[:, 0]
            assert (np.allclose(exp_data, sim, rtol=0.05))

        assert (np.allclose(proj.scale_factors['Variable_1'].sf, 3.75, rtol=0.05))

        project_residuals = proj.residuals(log_project_param_vector)
        # TODO: Add accuracy check

    def test_variable_jacobian(self):
        proj = TestProject.proj
        log_project_param_vector = TestProject.log_project_param_vector
        project_param_vector = np.exp(log_project_param_vector)
        proj.calc_project_jacobian(log_project_param_vector)

        model_jac = proj.get_model_jacobian_df()

        for experiment in proj.experiments:
            k_synt_idx = experiment.param_global_vector_idx['k_synt']
            k_deg_idx = experiment.param_global_vector_idx['k_deg']
            k_synt = project_param_vector[k_synt_idx]
            k_deg = project_param_vector[k_deg_idx]
            exp_t = experiment.get_variable_measurements('Variable_1').timepoints
            exp_t = exp_t[exp_t != 0]

            anal_jac = _simple_model_analytical_jac(k_deg, k_synt, exp_t)
            k_synt_analytical_jac = anal_jac[1, :]
            k_deg_analytical_jac = anal_jac[0, :]

            k_synt_sensitivity_jac = model_jac.loc[(experiment.name, 'Variable_1'), :].values[:, k_synt_idx]
            k_deg_sensitivity_jac = model_jac.loc[(experiment.name, 'Variable_1'), :].values[:, k_deg_idx]

            assert np.allclose(k_synt_analytical_jac, k_synt_sensitivity_jac, rtol=0.05)
            assert np.allclose(k_deg_analytical_jac, k_deg_sensitivity_jac, rtol=0.05)

    def test_calc_project_jacobian(self):
        proj = TestProject.proj

        project_param_vector = np.zeros((3,))
        low_deg_idx = proj.project_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj.project_param_idx['Group_1'][('High',)]
        synt_idx = proj.project_param_idx['k_synt']['Global']

        project_param_vector[high_deg_idx] = 0.01
        project_param_vector[low_deg_idx] = 0.001
        project_param_vector[synt_idx] = 0.01
        log_project_param_vector = np.log(project_param_vector)

        sens_jacobian = proj.calc_project_jacobian(log_project_param_vector)

        def get_scaled_sims(x):
            proj.residuals(x)
            sims = proj.get_simulations(scaled=True)
            return sims.values[:, 0]

        num_global_jac = approx_fprime(log_project_param_vector, get_scaled_sims, centered=True)

        assert np.allclose(num_global_jac, sens_jacobian, atol=0.000001)

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
        low_deg_idx = proj.project_param_idx['Group_1'][('Low',)]
        high_deg_idx = proj.project_param_idx['Group_1'][('High',)]
        synt_idx = proj.project_param_idx['k_synt']['Global']

        project_param_vector[high_deg_idx] = 0.01
        project_param_vector[low_deg_idx] = 0.001
        project_param_vector[synt_idx] = 0.01

        log_project_param_vector = np.log(project_param_vector)
        mod_param_vector = np.copy(log_project_param_vector)
        mod_param_vector[0] += 0.2

        proj.residuals(log_project_param_vector)
        old_scale_factor = proj.scale_factors['Variable_1'].sf
        # Changing param vector:
        proj.residuals(mod_param_vector)
        new_scale_factor = proj.scale_factors['Variable_1'].sf

        assert np.allclose(old_scale_factor, new_scale_factor, rtol=0.0001)

    def test_optimization(self):
        try:
            from leastsq_mod import leastsq as geo_leastsq
        except ImportError:
            from scipy.optimize import leastsq as geo_leastsq
            # Fallback
    
        proj = TestProject.proj

        base_guess = np.log(np.ones((3,)) * 0.01)

        out = geo_leastsq(proj.residuals, base_guess, Dfun=proj.calc_project_jacobian)

    @raises(KeyError)
    def test_remove_absent_experiment(self):
        proj = TestProject.proj
        proj.remove_experiments_by_settings({'Absent:', 5})

    def test_add_experiment(self):
        #############################################################################################
        # Set Up
        #############################################################################################
        proj = TestProject.proj
        exp_timepoints = np.linspace(0, 100, 10)
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

        if present == 1:
            raise AssertionError('Unable to remove experiment from project')

    def test_experiment_settings(self):
        raise AssertionError('Not Implemented Yet')

    def test_experiment_order(self):
        raise AssertionError('Not Implemented Yet')

    def test_priors_after_experiments(self):
        raise AssertionError('Not Implemented Yet')

    def test_sum_variables(self):
        #############################################################################################
        # Set Up
        #############################################################################################
        from scipy.integrate import odeint
        from test_utils.jittable_mm_model import model, ordered_params
        from test_utils.sens_jittable_mm_model import sens_model
        from test_utils.michelis_menten_model import michelis_menten

        try:
            from leastsq_mod import leastsq

            scipy_leastsq = False
        except ImportError:
            from scipy.optimize import leastsq

            scipy_leastsq = True
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

        param_dict = {'vmax': {'Global': vmax}, 'km': {'Global': km}, 'k_synt_s': {'Global': k_synt_s},
                      'k_deg_s': {'Global': k_deg_s}, 'k_deg_p': {'Global': k_deg_p}}
        sorted_param_vector = proj.project_param_dict_to_vect(param_dict)
        log_sorted_param_vector = np.log(sorted_param_vector)
        #############################################################################################

        """
        Testing since we didn't specify any parameter settings, all parameters got set as global
        """
        for param in proj.project_param_idx:
            setting = proj.project_param_idx[param].keys()[0]
            assert (setting == 'Global')
        #############################################################################################
        #############################################################################################

        """
        Testing to see that the residuals of the simulation, when using the same parameters of the ODE
        model used to generate it, are close to zero.
        """
        residuals = proj.residuals(log_sorted_param_vector)
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
            proj.residuals(pars)
            sims = proj.get_simulations()
            sim_vals = sims.loc[(slice(None), 'Total'), 'mean']
            return sim_vals

        num_jac = approx_fprime(log_sorted_param_vector, calc_all_sims, centered=True)
        sens_jac = proj.calc_project_jacobian(log_sorted_param_vector)
        assert np.allclose(sens_jac, num_jac, atol=0.00001)
        #############################################################################################
        random_params = np.random.randn(5).flatten()
        if scipy_leastsq:
            out = leastsq(proj.residuals, random_params, Dfun=proj.calc_project_jacobian, ftol=0.001,
                          maxfev=5000)
            fit_params = out[0]
        else:
            fit_params = leastsq(proj.residuals, random_params, jacobian=proj.calc_project_jacobian)

        proj.calc_sum_square_residuals(fit_params)
        all_sim = proj.get_simulations()

        fit_sim = all_sim.loc[(slice(None), 'Total'), 'mean']
        fit_t = all_sim.loc[(slice(None), 'Total'), 'timepoints']

        # Plotting
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.plot(fit_t, fit_sim, 'r-')
        ax.plot(t_sim, np.sum(sim, axis=1), 'ro')
        ax.set_title("Fitting Total")
        #############################################################################################

        # try fitting S & P separately
        substrate_measure = TimecourseMeasurement('Substrate', sim[:, 0], t_sim)
        product_measure = TimecourseMeasurement('Product', sim[:, 1], t_sim)

        substrate_experiment = Experiment('Substrate Experiment', [substrate_measure, product_measure])
        measurement_to_model_map = {'Substrate': ('direct', 0), 'Product': ('direct', 1)}

        mm_model = ode_model.OdeModel(model, sens_model, 2, ordered_params)
        sf_groups = [frozenset(['Substrate', 'Product'])]
        proj = Project(mm_model, [substrate_experiment], {}, measurement_to_model_map, sf_groups=sf_groups)

        proj.set_parameter_log_prior('k_synt_s', 'Global', np.log(0.01), 0.5)
        proj.set_parameter_log_prior('k_deg_s', 'Global', np.log(0.01), 0.5)
        proj.set_parameter_log_prior('k_deg_p', 'Global', np.log(0.01), 0.5)
        #proj.set_scale_factor_log_prior(frozenset(['Substrate', 'Product']), np.log(1.0), 0.1)
        # Strong prior around 1

        # NLopt Optimization
        random_params = np.random.randn(5).flatten()
        opt = nlopt.opt(nlopt.GN_CRS2_LM, proj.n_project_params)
        high_bounds = np.ones_like(random_params) * 5
        low_bounds = np.ones_like(random_params) * -5
        opt.set_upper_bounds(high_bounds)
        opt.set_lower_bounds(low_bounds)
        opt.set_min_objective(proj.nlopt_fcn)
        opt.set_maxtime(5)
        nlopt_params = opt.optimize(random_params)

        # Getting simulated values out.
        residuals = proj.residuals(nlopt_params)
        all_sim = proj.get_simulations(scaled=True)

        sub_sim = all_sim.loc[(slice(None), 'Substrate'), 'mean']
        sub_t = all_sim.loc[(slice(None), 'Substrate'), 'timepoints']

        prod_sim = all_sim.loc[(slice(None), 'Product'), 'mean']
        prod_t = all_sim.loc[(slice(None), 'Product'), 'timepoints']

        sf = proj.scale_factors

        assert (sf['Product'] is sf['Substrate'])
        # They are the same scale factor

        ax = fig.add_subplot(122)
        ax.plot(sub_t, sub_sim, 'r-', label='substrate')
        ax.plot(t_sim, sim[:, 0], 'ro')
        ax.plot(prod_t, prod_sim, 'b-', label='product')
        ax.plot(t_sim, sim[:, 1], 'bo')
        ax.set_title("Fitting Product and Substrate")
        ax.legend(loc='best')
        plt.show()

        out = proj.calc_sum_square_residuals(nlopt_params)
        res = proj.residuals(nlopt_params)
        rss = 1 / 2.0 * np.sum(res ** 2)

    def test_load_project(self):
        """
        Check that, for some fairly complex projects, when we load them, we still get the same values
        """
        import dill
        import os
        import sys

        project_path = os.path.join(os.getcwd(), 'tests', 'test_utils', 'loading_test')
        sys.path.insert(0, project_path)

        #############################################################################################
        projects = [
            '4528.0150_lstq_RSS_Fri Mar 13 10:07:18 2015.pickle',  # L2Loss_Coop_Complex_Settings_Fit_everything
            '10546.9520_lstq_RSS_Tue May  5 04:06:57 2015.pickle',  # LogLoss_Coop_Tetramer_Binding
            '11258.3391_lstq_RSS_Thu Apr 30 12:01:55 2015.pickle']  # LogLoss_Non_Coop_Tetramer_Binding

        for project_name in projects:
            idx = project_name.find('_')
            old_rss = float(project_name[:idx])

            p_fh = open(os.path.join(project_path, project_name), 'rb')
            project_dict = dill.load(p_fh)
            # Ok - it loaded correctly.  Now check that we get the same answer.

            proj = project_dict.pop('project')
            proj.reset_calcs()
            params = proj.project_param_dict_to_vect(project_dict)
            new_rss = proj.calc_sum_square_residuals(params)
            assert (np.allclose(old_rss, new_rss, rtol=0.001))
