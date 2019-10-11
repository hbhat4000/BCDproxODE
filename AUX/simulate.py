import numpy as np
from scipy.integrate import ode
import sys
sys.path.append('./ODEs/')
from fitzhugh_nagumo import *
from lotka_volterra import *
from lorenz96 import *
from lorenz import *
from rossler import *


def simulate(ODE_str, x0, true_param, end_t, dt, noise_var):
    '''
    This function creates clean states and noisy observations for the ODEs.

    Input:
      ODE_str: Name of the model as a string. Set ODE_str to
            'fitzhugh_nagumo' or 'lotka_volterra' or 'rossler' or 'lorenz96'
      x0: A d-dimensional list that contains initial state at time 0.
      true_param: A p-dimensional list that contains the true parameters of ODE.
      end_t: The final time of the simulation. The start time is 0.
      dt: The time interval between samples.
      noise_var: The variance of the Gaussian noise.
            The noise will be used in creating noisy observations.

    Output:
      X: T*d numpy array that contains the clean states.
      Y: T*d numpy array that contains the noisy observations.
      dt: Time interval between samples.
    '''

    ode_fun = eval(ODE_str + '_ode') # change the string into a function.

    d = x0.shape[0] # dimension of the states
    t0 = 0
    r = ode(ode_fun).set_integrator('dopri5').set_f_params(true_param)
    r.set_initial_value(x0, t0)
    T = int((end_t / dt) + 1) # number of observations
    X = np.empty((T, d)) # clean states
    X[0, :] = x0
    idx = 1
    while idx < T:
        r.integrate(r.t + dt)
        X[idx, :] = r.y
        idx = idx + 1

    # create noisy observations
    Y = X + (np.random.normal(0, noise_var, (T, d)))

    return X, Y, dt


def predict(init_state, end_t, dt, params,ODE_str,appr):
    '''
    We use this function to predict the states,
    using eq.(5) [Euler] or eq. (11) [multistep] of the paper.

    Input:
        init_state: d-dimensional initial state.
        end_t: The final time of the simulation. The start time is 0.
        dt: Time interval between the states.
        params: p-dimensional parameter \theta.
        ODE_str: Name of the model as a string. Set ODE_str to '
            fitzhugh_nagumo' or 'lotka_volterra' or 'rossler' or 'lorenz96'
        appr: This determines the type of discretization and takes one of the
            three strings: 1)"euler" for 1-step Euler,2) "ad2" for the
            2-step Adam-Bashforth, and 3) "ad3" for the 3-step Adam-Bashforth.

    Output:
        pX: The predicted states, a T*d numpy array
    '''

    T = int((end_t / dt) + 1)  # number of observations from time 0 to end_t
    ode_fun = eval(ODE_str + '_ode')  # change the string into a function.
    pX = np.zeros((T,init_state.shape[0]))  # Predicted states
    pX[0,:] = init_state  # initial state

    # Repeatedly apply equation (5) for the Euler or
    # eq.(11) for the general multi-step methods
    for i in range(1, T):
        if(appr == 'euler'):
            temp1 = ode_fun(0, pX[i - 1, :], params)
            pX[i, :] = pX[i - 1, :] + temp1 * dt
        elif(appr == 'ad2'):
            if (i == 1):
                temp1 = ode_fun(0, pX[i - 1, :], params)
                pX[i, :] = pX[i - 1, :] + temp1 * dt
            else:
                temp2 = ode_fun(0, pX[i - 1, :], params)
                pX[i, :] = pX[i - 1, :] + ((3 / 2) * (temp2 * dt))\
                           - ((1 / 2) * dt * (temp1))
                temp1 = temp2
        else:
            if (i == 1):
                temp1 = ode_fun(0, pX[i - 1, :], params)
                pX[i, :] = pX[i - 1, :] + temp1 * dt
            elif (i == 2):
                temp2 = ode_fun(0, pX[i - 1, :], params)
                pX[i, :] = pX[i - 1, :] + ((3 / 2) * (temp2 * dt)) - \
                           ((1 / 2) * dt * (temp1))
            else:
                temp3 = ode_fun(0, pX[i - 1, :], params)
                pX[i, :] = pX[i - 1, :] + ((23 / 12) * (temp3 * dt)) - \
                           ((4 / 3) * dt * (temp2)) + ((5 / 12) * dt * (temp1))
                temp1, temp2 = temp2, temp3
    return pX



