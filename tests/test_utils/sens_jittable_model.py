def sens_model(y, t, yout, p):

    #---------------------------------------------------------#
    #Parameters#
    #---------------------------------------------------------#

    k_deg = p[0]
    k_synt = p[1]


    #---------------------------------------------------------#
    #Variables#
    #---------------------------------------------------------#

    _y = y[0]


    #---------------------------------------------------------#
    #sensitivity Variables#
    #---------------------------------------------------------#

    sens_y_k_deg = y[1]
    sens_y_k_synt = y[2]


    #---------------------------------------------------------#
    #Differential Equations#
    #---------------------------------------------------------#

    yout[0] = (-_y*k_deg + k_synt)


    #---------------------------------------------------------#
    #sensitivity Equations#
    #---------------------------------------------------------#

    yout[1] = (-_y - k_deg*sens_y_k_deg)
    yout[2] = (-k_deg*sens_y_k_synt + 1)
