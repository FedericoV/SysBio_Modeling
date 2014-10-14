__author__ = 'Federico Vaggi'
import os

from symbolic import make_sensitivity_model


n_vars = 2


def michelis_menten(y, t, *args):
    p = args[0]

    #*! Parameters Start
    vmax = p[0]
    km = p[1]
    k_synt_s = p[2]
    k_deg_s = p[3]
    k_deg_p = p[4]
    #*! Parameters End


    #*! Variables Start
    _s = y[0]
    _p = y[1]

    #*! Variables End


    #*! Differential Equations Start
    d_s = - (vmax * (_s / (km + _s))) + k_synt_s - k_deg_s * _s
    d_p = vmax * (_s / (km + _s)) - k_deg_p * _p
    #*! Differential Equations End
    return [d_s, d_p]


if __name__ == '__main__':
    from scipy.integrate import odeint
    import numpy as np

    init_conditions = [0.01, 0]
    param_vector = np.array(
        [3e-4, 0.005, 0.001, 0.01, 0.01])  # Wiki http://en.wikipedia.org/wiki/Michaelis%E2%80%93Menten_kinetics
    # Data from Pepsin, assuming E_0 = 0.01
    t_sim = np.linspace(0, 100, 100)
    sim = odeint(michelis_menten, init_conditions, t_sim, args=(param_vector,))

    #plt.plot(t_sim, sim)
    #plt.legend(['Substrate', 'Product'])
    #plt.show()

    sens_jit_model = os.path.join(os.getcwd(), 'sens_jittable_mm_model.py')
    sens_jit_fh = open(sens_jit_model, 'w')
    mm_fh = open(os.path.realpath(__file__))
    make_sensitivity_model(mm_fh, sens_jit_fh)
    mm_fh.close()

    jit_model = os.path.join(os.getcwd(), 'jittable_mm_model.py')
    jit_fh = open(jit_model, 'w')

    mm_fh = open(os.path.realpath(__file__))
    make_sensitivity_model(mm_fh, jit_fh, calculate_sensitivities=False)
