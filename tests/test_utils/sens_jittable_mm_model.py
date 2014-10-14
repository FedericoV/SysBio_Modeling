def sens_model(y, t, yout, p):
    #---------------------------------------------------------#
    #Parameters#
    #---------------------------------------------------------#

    vmax = p[0]
    km = p[1]
    k_synt_s = p[2]
    k_deg_s = p[3]
    k_deg_p = p[4]


    #---------------------------------------------------------#
    #Variables#
    #---------------------------------------------------------#

    _s = y[0]
    _p = y[1]


    #---------------------------------------------------------#
    #sensitivity Variables#
    #---------------------------------------------------------#

    sens_s_vmax = y[2]
    sens_s_km = y[3]
    sens_s_k_synt_s = y[4]
    sens_s_k_deg_s = y[5]
    sens_s_k_deg_p = y[6]
    sens_p_vmax = y[7]
    sens_p_km = y[8]
    sens_p_k_synt_s = y[9]
    sens_p_k_deg_s = y[10]
    sens_p_k_deg_p = y[11]


    #---------------------------------------------------------#
    #Differential Equations#
    #---------------------------------------------------------#

    yout[0] = ((-_s * vmax + (_s + km) * (-_s * k_deg_s + k_synt_s)) / (_s + km))
    yout[1] = ((-_p * k_deg_p * (_s + km) + _s * vmax) / (_s + km))


    #---------------------------------------------------------#
    #sensitivity Equations#
    #---------------------------------------------------------#

    yout[2] = (-(_s * (_s + km) - sens_s_vmax * (_s * vmax + (_s + km) * (_s * k_deg_s - k_synt_s) - (_s + km) * (
    _s * k_deg_s + k_deg_s * (_s + km) - k_synt_s + vmax))) / (_s + km) ** 2)
    yout[3] = ((
               -_s ** 2 * k_deg_s * sens_s_km - 2 * _s * k_deg_s * km * sens_s_km + _s * vmax - k_deg_s * km ** 2 * sens_s_km - km * sens_s_km * vmax) / (
               _s ** 2 + 2 * _s * km + km ** 2))
    yout[4] = ((-sens_s_k_synt_s * (-_s * vmax - (_s + km) * (_s * k_deg_s - k_synt_s) + (_s + km) * (
    _s * k_deg_s + k_deg_s * (_s + km) - k_synt_s + vmax)) + (_s + km) ** 2) / (_s + km) ** 2)
    yout[5] = (-(_s * (_s + km) ** 2 - sens_s_k_deg_s * (
    _s * vmax + (_s + km) * (_s * k_deg_s - k_synt_s) - (_s + km) * (
    _s * k_deg_s + k_deg_s * (_s + km) - k_synt_s + vmax))) / (_s + km) ** 2)
    yout[6] = (sens_s_k_deg_p * (_s * vmax + (_s + km) * (_s * k_deg_s - k_synt_s) + (_s + km) * (
    -_s * k_deg_s - k_deg_s * (_s + km) + k_synt_s - vmax)) / (_s + km) ** 2)
    yout[7] = (_s ** 2 / (_s + km) ** 2 + _s * km / (_s + km) ** 2 - k_deg_p * sens_p_vmax + km * sens_s_vmax * vmax / (
    _s + km) ** 2)
    yout[8] = (-_s * vmax / (_s + km) ** 2 - k_deg_p * sens_p_km + km * sens_s_km * vmax / (_s + km) ** 2)
    yout[9] = (-k_deg_p * sens_p_k_synt_s + km * sens_s_k_synt_s * vmax / (_s + km) ** 2)
    yout[10] = (-k_deg_p * sens_p_k_deg_s + km * sens_s_k_deg_s * vmax / (_s + km) ** 2)
    yout[11] = (-_p - k_deg_p * sens_p_k_deg_p + km * sens_s_k_deg_p * vmax / (_s + km) ** 2)
