from scipy.optimize import minimize
import sys
sys.path.append('./ODEs/')
sys.path.append('./AUX/')
from fitzhugh_nagumo import *
from rossler import *
from lotka_volterra import *
from lorenz96 import *
from objectives import *
from autograd import grad
from simulate import *


def fit_direct(Y, dt, init_params, ODE_str, appr='euler', lam=1,
               max_iters=10000, tol=1e-8):
    '''
    This function learns the ODE parameters given the noisy observations.

    Input:
        Y: T*d numpy array that contains the noisy observations.
        dt: Time interval between the observations (states).
        init_params: p-dimensional initialization for the unknown parameters.
        ODE_str: This is a string, with the name of ODE model.
            ODE_str can be set to 'fitzhugh_nagumo' or 'lotka_volterra' or
            'rossler' or 'lorenz96'.
        appr: This determines the type of discretization and takes one of the
            three strings: 1)"euler" for 1-step Euler, 2) "ad2" for the 2-step
            Adam-Bashforth, and 3) "ad3" for the 3-step Adam-Bashforth
        lam: The hyper-parameter lambda in our method.
        max_iters: Maximum number of iterations.
        tol: Tolerance value to stop the optimization, if the amount of changes
            in the objective is small.
    Output:
        params: The estimated parameters.
        X: The estimated states.
        pred_X: The predicted states.
    '''

    # Initialization of states and parameters
    X = Y
    params = init_params

    # autograd accepts float data
    X = X.astype(float)
    params = params.astype(float)


    T, d = X.shape
    new_cost = 1000

    # autograd computes the derivative of the objectives automatically

    # This returns gradient of eq.(7) for the Euler or eq.(13a) for the
    #   multi-step method in the paper.
    param_grad = grad(param_obj)

    # This returns gradient of eq.(8) for the Euler or eq.(13b) for the
    # multi-step method in the paper.
    X_grad = grad(X_obj)

    # main loop of our algorithm
    for k in range(max_iters):

        # optimization over parameters given states
        res = minimize(param_obj, params, method='L-BFGS-B', jac=param_grad, args=(X, dt,appr,ODE_str),
               options={'disp': False,'maxcor': 100})
        params = res['x']

        # stop if the changes in the objective is smaller than tol
        prev_cost = new_cost
        new_cost = param_obj(params, X, dt,appr,ODE_str)
        if((prev_cost - new_cost) < tol and k > 1):
            break

        # print results every 500 iterations
        if(k%500 == 0):
            print('iter', k, 'params:', params, 'obj_val:', new_cost)

        # Flatten the current estimation and use it as initialization.
        X0 = X.flatten('F')
        # optimization over the states given the parameters
        res = minimize(X_obj, X0, method='L-BFGS-B', jac=X_grad,
                       args=(params, dt,X0+0.000001,lam,d,appr,ODE_str),
                           options={'disp': False,'maxcor': 100})
        x = res['x']
        X = x.reshape((d,T)).T #Put the data back into the original shape

    # predict the states using eq.(5) [Euler] or eq.(11) [multistep] of the paper
    pred_X = predict(X[0, :], (T-1)*dt, dt, params, ODE_str, appr)

    return params, X, pred_X



