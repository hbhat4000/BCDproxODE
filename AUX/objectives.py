import autograd.numpy as np
import sys
sys.path.append('./ODEs/')
from fitzhugh_nagumo import *
from lotka_volterra import *
from lorenz96 import *
from rossler import *


def X_obj(x, params, dt, x0, lam, d, appr, ODE_str):
    '''
    This is the objective function over the states X,
    in eq.(8) [Euler] or eq. (13b) [multistep] of the paper.

    Input:
        x: The Flattened current state X in eq(8) or eq(13b) of the paper.
        params: p-dimensional parameter \theta^*(n) in eq(8) or eq(13b) of the paper.
        dt: Time interval between the states.
        x0: The Flattened previous state X^*(n-1) in eq(8) or eq(13b) of the paper).
        lam: The hyperparameter lambda in our paper.
        d: dimension of the states.
        appr: This determines the type of discretization and takes one of the
            three strings: 1)"euler" for 1-step Euler, 2) "ad2" for the 2-step
             Adam-Bashforth, and 3) "ad3" for the 3-step Adam-Bashforth.
        ODE_str: Name of the model as a string. Set ODE_str to
                'fitzhugh_nagumo' or 'lotka_volterra' or 'rossler' or 'lorenz96'
    Output:
        objval: Return the objective value
    '''

    Td = x.shape[0] # Td = multiplication of T and d.
    T = int(Td / d)
    X = x.reshape((d,T)).T # Put X in the original T by d matrix

    # String to function. Given the current X and parameter,
    # it gives us the dX(t)/dt in eq.(1) of the paper.
    ode_vec = eval(ODE_str + '_ode_vec')
    terms = ode_vec(X,params)

    if(appr == 'euler'):
        # objective in eq(8) of the paper.
        objval = np.sum((X[1:T, :] - X[0:T - 1, :] - (terms*dt)) ** 2)
    else:
        # objective in eq(13b) of the paper, where the order is 3.
        temp = (X[1, :] - X[0, :] - (terms[0,:]*dt))
        objval = np.sum(temp ** 2)

        temp = (X[2, :] - X[1, :] - ((3/2)*(terms[1,:]*dt)))\
                + ((1/2)*terms[0,:]*dt)
        objval = objval + np.sum(temp ** 2)

        temp = (X[3:T, :] - X[2:T-1, :] - ((23 / 12) * (terms[2:T-1, :] * dt)))\
               + ((4 / 3) * terms[1:T-2, :] * dt) - ((5 / 12) * terms[0:T-3, :] * dt)
        objval = objval + np.sum(temp ** 2)

    # compute and return the objective value
    objval = objval + lam*np.sum((x - x0)**2)
    return objval


def param_obj(params, X, dt,appr,ODE_str):
    '''
     This is the objective function over the states X,
     in eq.(7) [Euler] or eq. (13a) [multistep] of the paper.

     Input:
         params: p-dimensional parameter \theta in eq(7) or eq(13a) of the paper.
         X: T*d matrix of the current states X^*(n-1) in eq(7) or eq(13a) of the paper.
         dt: Time interval between the states.
         appr: This determines the type of discretization and takes one of the
            three strings: 1)"euler" for 1-step Euler, 2) "ad2" for the 2-step
             Adam-Bashforth, and 3) "ad3" for the 3-step Adam-Bashforth.
         ODE_str: Name of the model as a string. Set ODE_str to
                'fitzhugh_nagumo' or 'lotka_volterra' or 'rossler' or 'lorenz96'

     Output:
         objval: Return the objective value of eq(7) or eq(13a)
     '''

    T = X.shape[0] # number of observations

    # String to function. Given the current X and parameter,
    # it gives us the dX(t)/dt in eq.(1) of the paper.
    ode_vec = eval(ODE_str + '_ode_vec')
    terms = ode_vec(X, params)


    if(appr == 'euler'):
        # objective in eq(7) of the paper.
        objval = np.sum((X[1:T, :] - X[0:T - 1, :] - (terms*dt)) ** 2)
    else:
        # objective in eq(13a) of the paper.
        temp = (X[1, :] - X[0, :] - (terms[0,:]*dt))
        objval = np.sum(temp ** 2)

        temp = (X[2, :] - X[1, :] - ((3/2)*(terms[1,:]*dt)))\
                + ((1/2)*terms[0,:]*dt)
        objval = objval + np.sum(temp ** 2)

        temp = (X[3:T, :] - X[2:T-1, :] - ((23 / 12) * (terms[2:T-1, :] * dt)))\
               + ((4 / 3) * terms[1:T-2, :] * dt) - ((5 / 12) * terms[0:T-3, :] * dt)
        objval = objval + np.sum(temp ** 2)

    return objval