__author__ = 'Federico Vaggi'

n_vars = 1


def simple_model(y, t, *args):
#*! Bound Arguments Start
    param_idx = args[0]
    conditions = args[1]
    param_vector = args[2]

#*! Parameters Start
    k_deg = param_vector[param_idx['k_deg']]
    k_synt = param_vector[param_idx['k_synt']]
#*! Parameters End
#*! Bound Arguments End


#*! Variables Start
    _y = y[0]
#*! Variables End


#*! Differential Equations Start
    d_y = k_synt - k_deg * _y
#*! Differential Equations End
    return d_y

if __name__ == '__main__':
    from scipy.integrate import odeint
    import numpy as np
    import matplotlib.pyplot as plt
    init_conditions = [0]
    param_idx = {'k_deg': 0, 'k_synt': 1}
    param_vector = np.array([0.001,  0.01])
    t_sim = np.linspace(0, 100, 100)
    low_deg = odeint(simple_model, init_conditions, t_sim, args=(param_idx, None, param_vector))
    t_exp = np.array([0.         ,  11.11111111,   22.22222222,   33.33333333,
                      44.44444444,  55.55555556,   66.66666667,   77.77777778,
                      88.88888889,  100.])
    t_idx = np.searchsorted(t_sim, t_exp)
    print t_sim[t_idx]
    print low_deg[t_idx].flatten()

    print '_____________________________________________________________________________________'

    param_vector = np.array([0.01,  0.01])
    high_deg = odeint(simple_model, init_conditions, t_sim, args=(param_idx, None, param_vector))
    plt.plot(t_sim, high_deg, 'r-')
    plt.plot(t_sim, low_deg, 'b-')

    t_idx = np.searchsorted(t_sim, np.arange(5, 95, 3.5))
    print t_sim[t_idx]
    print high_deg[t_idx].flatten()

